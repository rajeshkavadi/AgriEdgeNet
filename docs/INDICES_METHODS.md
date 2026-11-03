# Vegetation indices from the Parrot Sequoia: methods and rationale

Companion to `notebooks/src/02_sequoia_alignment_indices.py` (Part 2 of the combined notebook).
For each processing step, this document gives what the step does, why our data needs it, and what it does **not** guarantee.
Numbers quoted are from the 12-image flight (`IMG_260707_*`, 4 captures x NIR/RED/REG).

---

## 0. Summary

| Stage | Input -> output | Why it is needed here |
|---|---|---|
| 1. Metadata | TIF -> XMP + EXIF calibration fields | Every later step depends on per-image values (exposure, ISO, sensor model, vignetting, Sunshine). |
| 2. Sensor model | raw DN -> sensor-calibrated value *I* | Captures and bands were taken with different exposure/ISO; raw DN are not comparable. |
| 3. Vignetting | *I* -> *I* / *V(x,y)* | Edges of each frame are darker than the centre because of the lens, not the scene. |
| 4. Registration | 3 band images -> 1 pixel grid | The four lenses sit side by side; the same ground point lands on different pixels in each band. |
| 5. Illumination normalisation | *I* -> *I* x (E_ref / E_capture) | Sunlight changes between captures; without this, index differences between captures partly reflect the light, not the crop. |
| 6. (Optional) reflectance | *I* -> ρ via panel | Only route to absolute, cross-flight comparable values. **Not run: no panel image in this flight.** |
| 7. Indices | NDVI, NDRE per pixel | The quantities of interest. |
| 8. QC | registration scores, residuals, NDRE vs NDVI checks | Shows the outputs are physically plausible and properly aligned. |

**What the outputs are:** illumination-normalised, sensor-calibrated indices, valid for **relative comparison within this flight**.
**What they are not:** reflectance-based indices; they cannot be compared directly with other flights, other sensors or published thresholds.

---

## 1. Why NDVI and NDRE

Healthy leaves absorb strongly in red (chlorophyll absorption, ~660 nm) and reflect strongly in the near-infrared (leaf cell structure, ~790 nm). The red-edge band (~735 nm) sits on the steep slope between the two.

$$\mathrm{NDVI} = \frac{\mathrm{NIR}-\mathrm{RED}}{\mathrm{NIR}+\mathrm{RED}}, \qquad
\mathrm{NDRE} = \frac{\mathrm{NIR}-\mathrm{REG}}{\mathrm{NIR}+\mathrm{REG}}$$

* **NDVI** separates vegetation from soil and water well, but **saturates** in dense canopy: once red is almost fully absorbed, extra biomass barely changes it (typically above leaf-area index ~3).
* **NDRE** uses red-edge in place of red. Red-edge is only partly absorbed, so NDRE keeps responding to chlorophyll and nitrogen content in dense canopy, where NDVI has flattened.
* **NDRE is always lower than NDVI for vegetation.** Leaves reflect much more red-edge than red, so the NDRE denominator is larger and the numerator smaller. Typical green canopy: NDVI 0.6–0.9, NDRE 0.2–0.5. A pixel with NDRE > NDVI is a warning sign (misregistration, shadow, a non-vegetation surface or a calibration error), so the pipeline counts them (section 8).

For our use (crop vigour and nitrogen status in wheat, gram, potato, linseed and fenugreek), the two indices are complementary:

* NDVI for canopy cover and early growth.
* NDRE for canopy-closure stages, where NDVI is saturated.

---

## 2. Reading the calibration metadata

Each Sequoia TIF carries two metadata blocks:

* **EXIF** (standard photo tags): exposure time *t*, ISO, f-number *f*.
* **XMP** (Parrot's own XML block): `SensorModel` (A, B, C), `VignettingPolynomial2D` and `...Name`, `BandName`, camera yaw/pitch/roll, and `IrradianceList`, the base64-encoded Sunshine sensor log.

The reader takes each field from the block that holds it and **raises an error if a required field is missing**. The original notebook silently fell back to default values, which produces plausible-looking but wrong indices: the worst kind of failure.

`IrradianceList` is a sequence of 28-byte records, laid out as `<QHHHH3f`:

| Field | Meaning |
|---|---|
| timestamp_us | time of the Sunshine reading |
| ch0 | irradiance count in this band |
| ch1 | second channel (not used) |
| gain_index | Sunshine amplifier setting |
| integration_ms | Sunshine integration time |
| yaw, pitch, roll | Sunshine sensor orientation |

The irradiance proxy used is the median over records of ch0 / (gain x integration time).

---

## 3. Sensor model: raw DN to *I*

Raw pixel values (DN) depend on how the camera was set when it fired, not only on the scene. The Sequoia picks exposure time and ISO automatically per band and per capture, so in our 12 images the same leaf gives different DN in different frames. Parrot's sensor model (Parrot application note SEQ-AN-01) removes this:

$$I = \frac{f^2\,(\mathrm{DN}-B)}{A\,t\,\mathrm{ISO} + C}$$

| Term | Role |
|---|---|
| *B* | dark level: the DN the sensor outputs with no light (thermal and electronic offset); subtracting it sets the true zero |
| *A* | band sensitivity (gain): how many DN one unit of light produces per unit exposure x ISO |
| *C* | small offset in the exposure term |
| *t*, ISO | the camera's exposure settings for that frame |
| *f²* | aperture term: light reaching the sensor scales with 1/f², so multiplying by f² refers *I* to the light leaving the scene |

**What *I* is:** a quantity proportional to the radiance leaving the scene in that band, in Parrot's arbitrary units. Two frames of the same scene under the same light now give the same *I*, whatever exposure the camera chose.

**What *I* is not: reflectance.**

* Dark subtraction fixes the **zero point**; it does not fix the **scale**.
* Reflectance is a ratio, light leaving divided by light arriving (ρ = πL/E). *I* contains only the light leaving.
* The same wheat plot at noon and at 5 pm gives very different *I* but the same reflectance.
* Nor do A, B, C give absolute radiance: Parrot does not publish a conversion from *I* to W·m⁻²·sr⁻¹·nm⁻¹.

---

## 4. Vignetting correction

Every lens transmits less light toward the edge of the frame than at the centre ("vignetting"). Uncorrected, the corners of each band look darker, and because falloff differs per band and lens, it would also bias the band *ratios* used by NDVI/NDRE toward the frame edges.

Parrot stores a per-band 2-D polynomial on normalised coordinates *x* = col/W, *y* = row/H:

$$V(x,y) = \sum_k c_k\, x^{i_k} y^{j_k}, \qquad I \leftarrow I \,/\, V(x,y)$$

The powers (*i*, *j*) and coefficients *c* are read from the XMP of each image; nothing is hard-coded.

---

## 5. Band registration

### 5.1 Why it is needed

The Sequoia has four separate monochrome cameras, each with its own lens, sitting next to each other. Each band therefore sees the ground from a slightly different position and orientation, so a given ground point falls on different pixel coordinates in NIR, RED and REG.

NDVI is computed **pixel by pixel**. If NIR at pixel (r, c) comes from a leaf and RED at (r, c) comes from the soil beside it, the "NDVI" of that pixel is meaningless. The error is largest at boundaries (plot edges, rows, canopy gaps), exactly where crop structure is. In our data the raw offset from REG at the frame centre is about 19 px for NIR and 26 px for RED (from `alignment_summary.csv`), i.e. roughly 0.45 m and 0.6 m on the ground at the Sequoia's ~2.4 cm pixel. That is far too large to ignore.

### 5.2 Model: homography

A homography (8-parameter projective transform) maps one band onto another exactly when:

1. the scene is a plane, or
2. the cameras share a centre of projection.

Neither holds perfectly, but at 25 m above ground the small spacing between lenses makes the error from non-flat terrain (parallax) small.

**Worked estimate (assumed values).** Assume a lens spacing of about 2 cm, the 3.98 mm focal length and 3.75 µm pixel pitch of the mono cameras, and a 10 m-tall canopy over a ground plane at 25 m. The parallax is then about 0.6 px. For flat crop fields of under a metre in height it is far less than 0.1 px.

### 5.3 Why gradients and not raw intensities

The bands do not look alike:

* Vegetation is **dark in RED and bright in NIR**, while a road is bright in both, so the contrast flips sign between bands.
* Matching raw intensities (ordinary correlation) is pulled toward wrong solutions by this.
* The pipeline therefore registers **gradient-magnitude images**: Gaussian blur (σ = 1.5 px), then Sobel edges, then magnitude.
* Edges are in the same place in every band, even when their sign differs.

### 5.4 Why REG is the reference

Red-edge lies spectrally between RED and NIR, so its edge pattern resembles both. Registering RED -> REG and NIR -> REG gives two well-conditioned problems. Registering RED -> NIR directly would be the hardest pairing.

### 5.5 Algorithm

1. **Coarse start:** phase correlation at the coarsest level of a 4-level image pyramid estimates the translation. It is fast, robust to large offsets, and needs no starting guess. Both signs and the identity are tried, and the one with the best normalised cross-correlation is kept.
2. **Refinement:** ECC (enhanced correlation coefficient) optimises the full homography, coarse to fine (4 levels, up to 300 iterations per level, tolerance 1e-6). ECC is unaffected by brightness gain and offset differences between images, which suits multispectral bands.
3. **Consistency check across the flight:**
   * The lenses are rigidly mounted, so the band-to-band transform should be nearly the same for every capture at similar height.
   * The pipeline takes the per-band **median transform over the flight**.
   * Any capture whose ECC score is below 0.30, or whose transform moves the image corners or centre more than 3 px away from the median, is replaced by the median transform.
   * In this flight no capture needed the fallback; all deviated from the median by at most 1.63 px.
4. **Warping:** each band is resampled onto the REG grid (bilinear interpolation). Pixels that fall outside any band are set to NaN, and indices are computed only where all bands are valid.

**Tried and dropped:** lens (fisheye) undistortion before registration made the alignment worse (~40 px errors) with the stored coefficients, so it was removed. The homography absorbs most of the remaining distortion difference between lenses. Section 8 shows the residual is sub-pixel.

---

## 6. Illumination normalisation (Sunshine sensor)

### 6.1 Why it is needed

Between captures, the sun's height changes, thin cloud passes and haze varies. A 10% drop in sunlight lowers *I* in every band.

* The drop is not necessarily equal in all bands: diffuse skylight is bluer than direct sun, and cloud changes the spectral mix.
* So band ratios, and therefore NDVI, can shift between captures for reasons unrelated to the crop.

### 6.2 What the pipeline does

For each band, the Sunshine reading of each capture is compared with that band's **median over the flight**, and the image is rescaled:

$$I'_{band} = I_{band} \times \frac{\tilde E_{band}}{E_{band,\,capture}}$$

where Ẽ is the flight median. After this, every capture is expressed "as if taken under the flight's median light".

### 6.3 What it does not do

* **Does not produce reflectance.** The Sunshine counts are in their own units. Converting them to the camera's *I* units needs a per-band factor K that the metadata does not give for this flight (section 7).
* **Does not correct the Sunshine sensor's tilt.** The sensor tilts with the drone. Under clear sky, a tilt of ~10° toward or away from the sun can change its reading by several percent. Roll and pitch are recorded in each record but not used yet. Any tilt difference between captures becomes noise in this step.
* **Gain mapping unverified.** The `SUNSHINE_GAIN` table (gain index -> amplification) is probably not exact. It has no effect here because all captures used the same gain index; the pipeline prints a warning if the indices ever differ.

### 6.4 Effect on NDVI of a residual band-ratio error

If the NIR/RED ratio is off by a fraction *e* (from calibration, tilt or missing reflectance conversion), NDVI shifts by approximately

$$\Delta \mathrm{NDVI} \approx \frac{1-\mathrm{NDVI}^2}{2}\, e$$

Example: at NDVI = 0.6, a 10% ratio error moves NDVI by ~0.03. The bias is largest around NDVI ≈ 0 and smallest near ±1.

---

## 7. Reflectance: what is missing and how to get it

$$\rho_{band} = \frac{I_{band}}{K_{band}\, E_{band}}$$

Reflectance needs the product K·E per band under that flight's light. There are two routes:

| Route | How | Accuracy | Status |
|---|---|---|---|
| **Reflectance panel** | Photograph a calibrated panel (known ρ_p per band) just before or after the flight. Then K·E = I_panel / ρ_p, and each pixel's ρ = ρ_p · I / I_panel. | Limited by panel calibration and how much the light changes between the panel shot and the flight; the standard for publication. | **Not possible for this flight:** all 12 TIFs have `CalibrationPicture = 0`. The code supports it (`PANEL = {capture, roi, reflectance}`). |
| **Sunshine only** | Use the factory per-band calibration fields in the XMP to scale Sunshine counts to camera units (Pix4D's approach without a panel). | Approximate. My estimate is several percent absolute error; not validated on our data. | Not implemented; can be added as a flagged "approximate reflectance" mode and cross-checked against Pix4D. |

**Why reflectance matters for us:**

* Index thresholds (the illustrative NDVI 0.2 / 0.5 and NDRE 0.1 / 0.3 classes in the config), comparisons across dates, and comparisons with literature all assume reflectance.
* Without it, the present outputs support statements like "plot X is greener than plot Y in this flight".
* They do not support statements like "plot X has NDVI 0.62, indicating adequate nitrogen".

---

## 8. Quality control and current results

### 8.1 Registration quality

| Check | What it tells us | Result (this flight) |
|---|---|---|
| ECC score per band and capture | how well the gradient images match after warping (1 = perfect) | NIR 0.86–0.89; RED 0.36–0.45. All above the 0.30 floor. RED scores are lower because RED edges differ most from REG; use ECC as a pass/fail flag, not as accuracy. |
| Gradient NCC before vs after | improvement from registration | NIR 0.18–0.27 before -> 0.81–0.85 after; RED 0.08–0.11 -> 0.33–0.41 |
| Deviation from the flight-median transform | **consistency** between captures, not accuracy | 0.21–1.27 px (NIR), 0.28–1.63 px (RED) |
| **Residual misalignment** (local phase correlation on a 3x3 grid of patches after warping) | actual remaining offset, in pixels | NIR median 0.39 px (max 1.03); RED median 0.36 px (max 1.33) over 31 of 36 patches; 5 RED patches had too little texture to measure |

The residual check is the one that supports the claim **"sub-pixel alignment, worst case about 1.3 px"**. Consistency with the flight median alone would not, because all captures could share the same error. The residual check is currently a separate script; adding it to the notebook is pending.

### 8.2 Index plausibility

| Capture | NDVI median | NDRE median |
|---|---|---|
| 1 | 0.66 | 0.08 |
| 2 | 0.57 | 0.05 |
| 3 | 0.55 | 0.03 |
| 4 | 0.55 | 0.04 |

* Medians are over all valid pixels in the frame (vegetation plus road), so they describe the **scene**, not a crop.
* NDRE < NDVI throughout, as expected. On vegetation pixels, NDRE exceeds NDVI in at most 0.21% of pixels: a low rate, consistent with good registration.
* NDRE values are low in absolute terms. This is consistent both with the scene (tree canopy and road, not crops) and with the missing reflectance conversion (section 6.4). The current data cannot separate the two causes.

---

## 9. Limitations specific to our case

1. **Scene mismatch.** The 12 Sequoia frames show tree canopy over a road, not the five crops. The pipeline is validated on them for alignment and calibration only; no agronomic conclusion should be drawn from these numbers.
2. **No per-crop indices.** The RGB segmentation (Pi Camera v2) and the Sequoia images come from different cameras with no shared georeference. Per-crop NDVI/NDRE needs either:
   * a Sequoia flight over the plots, with segmentation run on the Sequoia's own registered bands or RGB, or
   * orthomosaics of both sensors on a shared coordinate system, with the masks rasterised onto the index grid.
3. **Not reflectance** (section 7).
4. **Uncorrected effects:**
   * Sunshine sensor tilt;
   * temperature drift of dark level and gain (partly covered by the per-image B);
   * the XMP `InvalidPixel` list (not masked yet);
   * direction-dependent reflectance (leaves look brighter or darker depending on sun and view angle, which biases frame edges).
5. **GRE band** is not present in this set, so green-based indices cannot be computed. False-colour QC uses REG in its place.

---

## 10. Recommended protocol for the next flight

1. **Panel shot** before and after the flight, at the flight's height or handheld at about 1 m, with no shadow on the panel. This gives reflectance and removes the largest uncertainty.
2. **Fly over the crop plots** at the RGB height (25 m) so masks and indices can be joined.
3. Fly in **stable light** (clear sky or uniform overcast), within ±2 h of solar noon, to limit sun-angle and Sunshine-tilt effects.
4. Record all four bands, including GRE, for a green-based index and better false-colour QC.
5. **Process the same images in Pix4D (or Agisoft) as an independent check.** Agreement within ~0.02 NDVI supports the pipeline; disagreement points to the calibration step.

---

## 11. Output files

| Path (under `Sequoia_Indices/`) | Contents |
|---|---|
| `Calibrated_Bands/<capture>_bands.tif` | aligned, calibrated, illumination-normalised bands (float32; NaN outside the shared area); `radiometric_level` tag records the processing level |
| `NDVI/`, `NDRE/` | per-capture index GeoTIFF-style float images and colour-mapped PNGs |
| `Alignment_QC/<capture>_alignment_QC.png` | false colour before/after registration (R = NIR, G = RED, B = REG). Coloured fringes at edges before registration should disappear after it. |
| `alignment_summary.csv` | per band and capture: method, ECC score, NCC before/after, centre shift, deviation from flight median, homography |
| `indices_summary.csv` | per capture: processing level, valid area, NDVI/NDRE mean, median, 5th and 95th percentiles, class shares |
| `PROCESSING_REPORT.txt` | settings and per-capture summary in plain text |
