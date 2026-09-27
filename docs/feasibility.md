# Reading Nepali plates at GCM Gate 1

I measured what the two existing cameras actually deliver before doing any
model work, because no amount of fine tuning recovers detail the sensor never
captured. This is what I found.

## The short version

The current cameras cannot support plate recognition, and this is an optics
problem rather than a model problem. Vehicle detection, direction, and entry
and exit timing all work fine on the same footage. If plate numbers are a hard
requirement, the fix is a third camera, not a better network.

## What I measured

I pulled two plates out of `busy_day.jpg`, a frame from the exit camera and the
most favourable frame in the sample set. Daylight, vehicles parked and close,
no motion blur.

| Vehicle | Plate size in frame | Character height |
|---|---|---|
| Toyota Hiace, two row plate | 53 x 51 px | roughly 12 to 15 px |
| Green tourist bus, single row | 87 x 21 px | roughly 15 px |

Reliable recognition wants 30 px of character height for Devanagari. Latin
plates get away with 20 px, but Devanagari needs more, because the detail that
separates ३ from ४, ६ from ७, and ब from व is finer than any distinction Latin
script asks a recogniser to make.

## What OCR does with that

I ran EasyOCR against those exact crops, with both the Devanagari and English
models, at 1x, 4x, and 8x upscaling.

```
hiace     (truth ≈ बा • च ••••)   →  "इ२00"    conf 0.05
                                   →  "37od"    conf 0.16
greenbus                           →  "तःष"     conf 0.03
                                   →  "(23e475" conf 0.04
```

Every read is wrong and nothing clears 0.16 confidence. Upscaling changes the
output but never improves it, which is the signature of missing information
rather than a weak model.

Worth flagging separately, `main.py:13` in v1 constructs
`easyocr.Reader(['en'])` while the comment claims Devanagari support. v1 has
never been able to read a Nepali plate, so any plate column it produced is
noise.

## Why, in numbers

`resolution_budget.py` works this out from the camera spec. The UNV
IPC2225SE-DF40K-WL-I0 gives 2880 px across an 80.2 degree horizontal field,
so its focal length is about 1710 px. A plate character is about 7 cm tall,
which gives a single governing equation.

```
max_distance_m = 1710 * 0.07 / required_character_pixels
```

| Required character height | Furthest the plate can be |
|---|---|
| 30 px, reliable | 4.0 m |
| 20 px, marginal | 6.0 m |
| 15 px, what we have | 8.0 m |

The readable zone for this camera ends at 4 to 6 metres. Both cameras are
mounted high on the gate and cover 15 to 25 metres of scene. That gap is the
whole problem.

The model checks out against reality. Back solving the 53 px Hiace plate puts
it at 9.7 m from the camera, which matches where it sits in the frame, and
predicts 12 px characters, which matches what I measured by hand.

Two further losses stack on top and both get worse at a gate. Mounting high
means looking down at a plate that faces the horizon, and that obliquity costs
another factor of cos(angle) on effective resolution. Moving vehicles at a
1/30 s shutter add motion blur, and at night the ColorHunter mode opens the
shutter further.

## What fine tuning can and cannot buy

Fine tuning is not worthless here, and I want to be precise about where the
line falls.

It can plausibly move the usable threshold from 30 px down to somewhere near
18 to 20 px of character height, for three reasons that compound.

A plate specific recogniser beats a document OCR model badly. EasyOCR's
Devanagari model was trained on printed text, and embossed plate glyphs at low
resolution are a different distribution. Training on plate crops with
aggressive downscaling augmentation is the single biggest model side gain
available.

Constrained decoding removes most remaining errors for free. A plate is not an
arbitrary string. There are 14 legal zone tokens and 12 legal class letters,
in a fixed order, with bounded digit runs. `nepali_plate.py` implements this,
and forcing the output to be a plate that could legally exist eliminates a
large fraction of character level mistakes without touching the model.

Multi frame fusion is worth more than any single frame improvement. A vehicle
crossing the gate is visible for tens of frames at varying range and angle, and
the errors are largely independent between frames. Decoding every frame and
aggregating beats picking one good shot, which is what commercial ANPR does.
`fuse()` implements this.

What none of that buys is a plate at 87 x 21 px on the far side of the
courtyard. Three techniques that each need roughly 20 px still need roughly
20 px. Published Nepali ANPR results in the 93 to 96 percent range are real,
but as far as I can tell they are measured on close, static, roughly frontal
captures from Kaggle datasets rather than on overhead CCTV at range, so I would
not expect those numbers to transfer to this mounting. I have not verified
their capture conditions directly.

## The signal that does survive

Plate background colour encodes the ownership category under the pre 2020
system, and colour holds up at low resolution far better than glyph shape,
because it is hundreds of pixels of flat colour rather than fine stroke detail.

| Colour | Ownership |
|---|---|
| Red | Private |
| Black | Public |
| White | Government |
| Yellow | National corporation |
| Green | Tourist |
| Blue | Diplomatic |

This is readable at the resolution we already have, and for GCM's actual
question it carries real information. A black plate is a public vehicle, which
is what a van running students to the college looks like. The Hiace in the
sample frame has a black plate, consistent with exactly that.

I wired colour into the decoder as a prior. Passing an observed colour narrows
the class slot before decoding, which is the cheapest accuracy win in the whole
pipeline and costs one flat region classifier.

## What I would actually do

Ranked by value for effort.

Add one dedicated ANPR camera. Mounted low at roughly plate height, aimed along
the lane, narrow field of view, fast shutter, capturing vehicles at 3 to 5 m.
This is the only change that makes plate numbers genuinely reliable, and it
turns a research problem back into an engineering one. The existing two cameras
keep doing detection and counting, which they are well suited to.

Enrol the fleet instead of reading it. GCM's traffic is mostly a small,
recurring set of student transport vehicles. Photographing each one's plate
once at close range and matching thereafter on appearance is far more tractable
than reading plates at range every time, and it is more accurate than any OCR
would be. This handles the real use case without solving the general problem.

Classify plate colour now. Cheap, robust at current resolution, and useful on
its own.

Treat plate OCR as best effort and never as ground truth. Gate on confidence,
store low confidence reads as candidates for review rather than as facts, and
surface the ranked alternatives the decoder already produces.

## On the questions in `questions/difficultQuesions.md`

Two of those are answered directly by this work. A plate that does not exist
can no longer be logged, because `parse_plate` rejects any string that is not
structurally legal, and the decoder can only ever emit a plate that could
exist. The EasyOCR outputs above are in the test suite as cases that must be
rejected. A misread plate is handled by confidence gating and multi frame
agreement, so a disagreeing or weak read is reported as no read rather than as
a wrong one.

The format change question is real and I have partly hit it already. Nepal has
been rolling out embossed plates using Latin characters and province codes
alongside the older Devanagari zone plates. This module implements the pre 2020
system, which is most of what appears in the GCM footage. I did not implement
the new format because I am not confident enough in its exact structure from
what I could find, and guessing at a grammar would defeat the purpose of having
one. That needs a definitive source before it goes in.

## What I need next

Recorded clips of vehicles actually crossing the gate, which you said you can
pull. Stills of parked vehicles cannot tell me the read rate for a moving
vehicle, and that is the number that decides this. With clips I can measure
achievable accuracy properly instead of estimating it, and populate the
character confusion pairs from real data rather than assuming them.

Also useful, the exact SKU of the deployment machine, since the architecture
review notes Quick Sync decode is free performance if the i9 has integrated
graphics.

## Files

`resolution_budget.py` computes the optical limit for any camera and geometry,
and back solves range from an observed plate. Run it directly to print the
table above.

`nepali_plate.py` is the plate grammar, structural validation, colour aware
constrained decoding, and multi frame fusion. Pure stdlib, no model dependency,
so it sits behind EasyOCR now and a fine tuned recogniser later.

`test_nepali_plate.py` covers all of it, including the real EasyOCR failures as
rejection cases. 38 tests, all passing.

## Sources

- [Vehicle registration plates of Nepal](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Nepal)
- [Character Recognition of Nepali Number Plate](https://arxiv.org/pdf/2606.28946)
- [An Approach of Devanagari License Plate Detection and Recognition Using Deep Learning](https://link.springer.com/chapter/10.1007/978-3-030-88244-0_9)
- [Nepali License Plate Recognition with Deep Learning](https://sanjayasubedi.com.np/deeplearning/nepali-license-plate-recognition-with-deep-learning/)
- [Uniview IPC2225SE-DF40K-WL-I0 specifications](https://www.bhphotovideo.com/c/product/1646069-REG/uniview_ipc2225se_df40k_wl_i0_5mp_hd_colorhunter_fixed.html)
