#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include "esp_log.h"
#include "driver/i2s_std.h"
#include "usb_device_uac.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/portmacro.h"

static const char *TAG = "i2s_usb_asrc_cubic";

#define SAMPLE_RATE   48000

// I2S format (ICS-43434: 24-bit in 32-bit slot)
#define SAMPLE_BITS   I2S_DATA_BIT_WIDTH_32BIT
#define CHANNELS      2   // stereo

// USB format (16-bit stereo)
#define USB_CHANNELS          2
#define USB_BITS_PER_SAMPLE   16
#define USB_BYTES_PER_SAMPLE  (USB_BITS_PER_SAMPLE / 8)

// -----------------------------------------------------------------------------
// I2S handle / gain
// -----------------------------------------------------------------------------
static i2s_chan_handle_t s_rx_chan = NULL;
static float s_gain = 1.0f;   // host volume → linear gain

// -----------------------------------------------------------------------------
// RING BUFFER: 32-bit stereo frames (raw I2S data)
// -----------------------------------------------------------------------------

// Must be power-of-two for masking
#define RING_FRAMES   4096
#define RING_MASK     (RING_FRAMES - 1)

typedef struct {
    int32_t data[RING_FRAMES * CHANNELS]; // interleaved L,R,L,R...
    size_t  write_idx;                    // in frames
    size_t  read_idx;                     // in frames
    size_t  level;                        // number of frames stored
} audio_ring_t;

static audio_ring_t s_ring;
static portMUX_TYPE s_ring_mux = portMUX_INITIALIZER_UNLOCKED;

// -----------------------------------------------------------------------------
// ASRC state (cubic interpolation)
// -----------------------------------------------------------------------------

// input position in frames relative to ring.read_idx
static float s_src_phase = 0.0f;          // fractional position [0,1)
static float s_src_step  = 1.0f;          // input frames per output frame
static float s_fill_avg  = (float)RING_FRAMES / 2.0f;  // smoothed fill level

// -----------------------------------------------------------------------------
// RING BUFFER HELPERS
// -----------------------------------------------------------------------------

static void ring_push_frames(const int32_t *src, size_t frames)
{
    taskENTER_CRITICAL(&s_ring_mux);

    for (size_t f = 0; f < frames; f++) {
        size_t w = s_ring.write_idx & RING_MASK;

        s_ring.data[2 * w + 0] = src[2 * f + 0];
        s_ring.data[2 * w + 1] = src[2 * f + 1];

        s_ring.write_idx = (s_ring.write_idx + 1) & RING_MASK;

        if (s_ring.level < RING_FRAMES) {
            s_ring.level++;
        } else {
            // buffer full → drop oldest
            s_ring.read_idx = (s_ring.read_idx + 1) & RING_MASK;
        }
    }

    taskEXIT_CRITICAL(&s_ring_mux);
}

static inline void ring_snapshot(size_t *read_idx, size_t *level)
{
    taskENTER_CRITICAL(&s_ring_mux);
    *read_idx = s_ring.read_idx;
    *level    = s_ring.level;
    taskEXIT_CRITICAL(&s_ring_mux);
}

static inline void ring_consume(size_t frames)
{
    if (frames == 0) return;

    taskENTER_CRITICAL(&s_ring_mux);
    if (frames > s_ring.level) {
        frames = s_ring.level;
    }
    s_ring.read_idx = (s_ring.read_idx + frames) & RING_MASK;
    s_ring.level   -= frames;
    taskEXIT_CRITICAL(&s_ring_mux);
}

// -----------------------------------------------------------------------------
// 24-bit → 16-bit helper, reading from ring at relative frame index
// -----------------------------------------------------------------------------

static inline float ring_get_sample16_rel(size_t read_idx0,
                                          int rel_frame_index,
                                          int channel)
{
    // channel: 0 = L, 1 = R
    size_t idx = (read_idx0 + (size_t)rel_frame_index) & RING_MASK;
    int32_t raw = s_ring.data[2 * idx + (size_t)channel];

    // Extract 24-bit signed from left-aligned 32-bit
    uint8_t *b = (uint8_t *)&raw;

    int32_t s24 = ((int32_t)b[3] << 24) |
                  ((int32_t)b[2] << 16) |
                  ((int32_t)b[1] << 8);

    // downconvert 24 → 16 with some headroom (same as your old code)
    s24 >>= 12;

    return (float)s24;  // will be scaled/amped later
}

// -----------------------------------------------------------------------------
// USB CALLBACKS
// -----------------------------------------------------------------------------

static esp_err_t uac_output_cb(uint8_t *buf, size_t len, void *arg)
{
    (void)buf;
    (void)len;
    (void)arg;
    // capture-only
    return ESP_OK;
}

static void uac_set_volume_cb(uint32_t volume, void *arg)
{
    (void)arg;
    int16_t db_x256 = (int16_t)volume;
    float dB = db_x256 / 256.0f;
    s_gain = powf(10.0f, dB / 20.0f);
    ESP_LOGI(TAG, "Gain -> %.3f (%.2f dB)", s_gain, dB);
}

// -----------------------------------------------------------------------------
// Cubic interpolation helper (Catmull-Rom style)
// y0,y1,y2,y3 are float sample values, t in [0,1]
// -----------------------------------------------------------------------------

static inline float cubic_interp(float y0, float y1, float y2, float y3, float t)
{
    float a0 = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
    float a1 =        y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
    float a2 = -0.5f * y0 + 0.5f * y2;
    float a3 =        y1;
    return ((a0 * t + a1) * t + a2) * t + a3;
}

// -----------------------------------------------------------------------------
// USB INPUT CALLBACK (USB pulls audio)
// ASRC with cubic interpolation, USB = master clock
// -----------------------------------------------------------------------------

static esp_err_t uac_input_cb(uint8_t *buf, size_t len, size_t *bytes_read, void *arg)
{
    (void)arg;

    if (!buf || len == 0) {
        if (bytes_read) *bytes_read = 0;
        return ESP_OK;
    }

    const size_t bytes_per_frame = USB_CHANNELS * USB_BYTES_PER_SAMPLE; // 4 bytes/stereo frame
    size_t frames_req = len / bytes_per_frame;
    int16_t *out = (int16_t *)buf;

    if (frames_req == 0) {
        if (bytes_read) *bytes_read = 0;
        return ESP_OK;
    }

    // Snapshot ring state once per callback
    size_t read_idx0, level0;
    ring_snapshot(&read_idx0, &level0);

    // Update smoothed fill level for drift estimation
    if (s_fill_avg <= 0.0f) {
        s_fill_avg = (float)level0;
    } else {
        s_fill_avg = 0.999f * s_fill_avg + 0.001f * (float)level0;
    }

    // Compute small step adjustment based on buffer fill error
    const float target_fill = (float)RING_FRAMES * 0.5f;
    float err = s_fill_avg - target_fill;

    // Kp chosen to keep adjustment ~ ±300ppm for large error
    const float Kp = 1.0e-7f;
    float new_step = 1.0f + Kp * err;

    // Clamp step to a sane range
    if (new_step < 0.9995f) new_step = 0.9995f;
    if (new_step > 1.0005f) new_step = 1.0005f;

    float step  = new_step;
    float phase = s_src_phase;

    size_t produced = 0;

    // If we have very little data, just output silence
    if (level0 < 4) {
        for (size_t i = 0; i < frames_req * USB_CHANNELS; i++) {
            out[i] = 0;
        }
        if (bytes_read) {
            *bytes_read = frames_req * bytes_per_frame;
        }
        return ESP_OK;
    }

    // We need at least 4 frames ahead for cubic; keep 3-frame safety margin
    int safe_frames = (int)level0 - 3;
    if (safe_frames < 1) safe_frames = 1;

    while (produced < frames_req) {
        // If we're too close to the end of available input, stop and pad with zeros
        if (phase + 1.0f >= (float)safe_frames) {
            break;
        }

        int idx = (int)phase;
        float t = phase - (float)idx;

        float outL_f, outR_f;

        // Boundary handling: near start or end, use linear instead of cubic
        if (idx < 1 || (idx + 2) >= (int)level0) {
            // Linear interpolation between idx and idx+1
            float L0 = ring_get_sample16_rel(read_idx0, idx,     0);
            float L1 = ring_get_sample16_rel(read_idx0, idx + 1, 0);
            float R0 = ring_get_sample16_rel(read_idx0, idx,     1);
            float R1 = ring_get_sample16_rel(read_idx0, idx + 1, 1);

            outL_f = L0 + (L1 - L0) * t;
            outR_f = R0 + (R1 - R0) * t;
        } else {
            // Cubic interpolation using idx-1, idx, idx+1, idx+2
            float Lm1 = ring_get_sample16_rel(read_idx0, idx - 1, 0);
            float L0  = ring_get_sample16_rel(read_idx0, idx,     0);
            float L1  = ring_get_sample16_rel(read_idx0, idx + 1, 0);
            float L2  = ring_get_sample16_rel(read_idx0, idx + 2, 0);

            float Rm1 = ring_get_sample16_rel(read_idx0, idx - 1, 1);
            float R0  = ring_get_sample16_rel(read_idx0, idx,     1);
            float R1  = ring_get_sample16_rel(read_idx0, idx + 1, 1);
            float R2  = ring_get_sample16_rel(read_idx0, idx + 2, 1);

            outL_f = cubic_interp(Lm1, L0, L1, L2, t);
            outR_f = cubic_interp(Rm1, R0, R1, R2, t);
        }

        // Apply gain
        outL_f *= s_gain;
        outR_f *= s_gain;

        // Clamp to int16
        if (outL_f >  32767.0f) outL_f =  32767.0f;
        if (outL_f < -32768.0f) outL_f = -32768.0f;
        if (outR_f >  32767.0f) outR_f =  32767.0f;
        if (outR_f < -32768.0f) outR_f = -32768.0f;

        out[2 * produced + 0] = (int16_t)outL_f;
        out[2 * produced + 1] = (int16_t)outR_f;

        produced++;
        phase += step;
    }

    // How many input frames did we consume?
    size_t adv = (size_t)phase;
    phase -= (float)adv; // keep fractional remainder

    // Commit consumption to ring
    ring_consume(adv);

    // Save ASRC state
    s_src_phase = phase;
    s_src_step  = step;

    // Pad any remaining requested frames with silence
    while (produced < frames_req) {
        out[2 * produced + 0] = 0;
        out[2 * produced + 1] = 0;
        produced++;
    }

    if (bytes_read) {
        *bytes_read = frames_req * bytes_per_frame;
    }

    return ESP_OK;
}

// -----------------------------------------------------------------------------
// I2S CAPTURE TASK
// Continuously reads from I2S and pushes into ring buffer.
// -----------------------------------------------------------------------------

static void i2s_capture_task(void *arg)
{
    (void)arg;

    // 256 stereo frames of 32-bit
    int32_t i2s_buf[256 * CHANNELS];

    while (1) {
        size_t bytes_read = 0;

        esp_err_t err = i2s_channel_read(
            s_rx_chan,
            (void *)i2s_buf,
            sizeof(i2s_buf),
            &bytes_read,
            portMAX_DELAY
        );

        if (err != ESP_OK || bytes_read == 0) {
            // Just log if you run this without USB connected
            ESP_LOGW(TAG, "i2s_channel_read err=%d bytes=%u",
                     (int)err, (unsigned)bytes_read);
            continue;
        }

        size_t frames = bytes_read / (sizeof(int32_t) * CHANNELS);
        ring_push_frames(i2s_buf, frames);
    }
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------

void app_main(void)
{
    ESP_LOGI(TAG, "Starting ICS-43434 I2S → USB (ASRC cubic)");

    memset(&s_ring, 0, sizeof(s_ring));
    s_src_phase = 0.0f;
    s_src_step  = 1.0f;
    s_fill_avg  = (float)RING_FRAMES / 2.0f;

    // ----- I2S INIT -----
    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);

    chan_cfg.auto_clear = true;

    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &s_rx_chan));

    i2s_std_clk_config_t clk_cfg = {
        .sample_rate_hz = SAMPLE_RATE,
        .clk_src        = I2S_CLK_SRC_DEFAULT,
        .mclk_multiple  = I2S_MCLK_MULTIPLE_256,
    };

    i2s_std_slot_config_t slot_cfg = {
        .data_bit_width = SAMPLE_BITS,
        .slot_bit_width = SAMPLE_BITS,
        .slot_mode      = I2S_SLOT_MODE_STEREO,
        .ws_width       = SAMPLE_BITS,
        .ws_pol         = false,
        .bit_shift      = true,   // Philips format
        .left_align     = true,
        .big_endian     = false,
        .slot_mask      = I2S_STD_SLOT_BOTH,
    };

    i2s_std_gpio_config_t gpio_cfg = {
        .mclk = I2S_GPIO_UNUSED,
        .bclk = 9,   // your wiring
        .ws   = 8,   // your wiring
        .dout = I2S_GPIO_UNUSED,
        .din  = 10,  // your wiring
    };

    i2s_std_config_t std_cfg = {
        .clk_cfg  = clk_cfg,
        .slot_cfg = slot_cfg,
        .gpio_cfg = gpio_cfg
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx_chan));

    // ----- START I2S CAPTURE TASK -----
    BaseType_t ok = xTaskCreatePinnedToCore(
        i2s_capture_task,
        "i2s_capture_task",
        4096,
        NULL,
        5,
        NULL,
        0
    );
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create i2s_capture_task");
    }

    // ----- USB INIT -----
    uac_device_config_t cfg = {
        .input_cb      = uac_input_cb,
        .output_cb     = uac_output_cb,
        .set_mute_cb   = NULL,
        .set_volume_cb = uac_set_volume_cb,
        .cb_ctx        = NULL,
    };

    ESP_ERROR_CHECK(uac_device_init(&cfg));

    ESP_LOGI(TAG, "USB audio interface READY (16-bit, 48kHz, stereo).");
}
