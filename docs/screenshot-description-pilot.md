# Screenshot Description Pilot

Status: pilot complete on a partial sample (6 old/new pairs). **Nothing was written to
Weaviate.** The full re-description is not yet run.

Date of measurements: 2026-11-09, against the hosted `SupportChunk` collection with the
read-only viewer key.

## Why this pilot was run

During the Epic 11 (content sync) investigation of how chunk UUIDs identify content, a
read-only inspection of the hosted collection found that the screenshot-description
chunks are the weakest part of the data.

### What is in the collection

| Measure | Result |
|---|---|
| Objects | 922 (708 web, 214 video), all 768-dimensional vectors |
| Source pages | 185 distinct `source_url` values; `url_hash` is the SHA-256 of `source_url` on all 922 |
| Ingested | 2026-03-13 to 2026-03-16 |
| Screenshot-description chunks | 381 (41% of the collection, 54% of web chunks) |

### Problem 1: screenshot chunks have no provenance

The other chunk types have UUIDs that can be recomputed from stored data:

| Scheme | Objects | Key |
|---|---|---|
| Web text | 327 | `source_url \| chunk_index` |
| Video (fused audio + visual) | 214 | `source_url \| fused \| start_seconds` |
| Web screenshot | 381 | `uuid5(image_url)`, but **`image_url` is not stored** |

For the 381 screenshot chunks nothing links the chunk to its image: no `image_url`, no
alt text, no position on the page, and an empty `section_heading`. The only record of the
image is one LLM-written description. Consequences:

- A sync cannot tell which chunk belongs to which image, so a replaced screenshot (new
  CDN URL, new UUID) leaves the old description in place forever.
- The description cannot be audited or regenerated from the stored data.
- The UI cannot show the screenshot next to a citation.

A test on one page showed that the mapping can be recovered by re-deriving
`uuid5(image_url)` from the live page's image URLs (see results below), so a backfill is
feasible.

### Problem 2: three runaway descriptions

The vision step (`describe_image`) has no output cap and no repetition check. Three
chunks are degenerate loops from the vision model:

| Chunk | Size | Content |
|---|---|---|
| `youtu.be/u62q7SaznFI`, window at 60 s | 229,225 characters | about 3 KB of real transcript, then "A square with a smiling face / cast icon / subtitles icon…" (one phrase 491 times) |
| HydroMan `article/268` (cut-outs) | 86,434 characters | a calculator screenshot enumerating `"2530", "2540", …, "13370"` |
| HydroMan `article/281` (sandwich plate) | 81,373 characters | a UI description looping "A symbol resembling a pressure gauge / bus / ground plane…" |

A single such chunk can consume the whole 10k-token context budget and push every later
source out of the prompt.

### Problem 3: the descriptions rarely help and sometimes hurt

Retrieval measurement (57 synthetic queries, top 10 each, the app's own `retrieve()`):

| Measure | Result |
|---|---|
| Screenshot chunks as a share of retrieved chunks | 3.2% (18 of 570), against 41% of the collection |
| Queries with at least one in the top 10 | 6 of 57 |
| Page-title questions / hand-written UI questions | 0.4% / 13.3% |
| Rank | mostly 4 to 10 |
| When retrieved, the page's only chunk in that answer | 16 of 18 |

With/without answer comparison on the 12 hand-written UI questions (`gemini-3.6-flash`,
one run per variant): screenshot chunks reached the top 10 for 4 questions.

| Question | Effect of screenshot chunks |
|---|---|
| Print the bill of materials | neutral |
| Grid snap toolbar icon | neutral |
| Where is the library manager window | worse: License manager, Open File and Library Setup descriptions crowded out the useful Library-panel chunk, and the answer said "no information" |
| Sandwich plate dialog | mildly worse: the answer drifted to an unrelated "Confirm dialog" layout |

None improved. The descriptions match on UI wording ("manager window", "dialog") rather
than on the page's topic, because nothing in the description says what the page is about.

Limits of this evidence: synthetic queries (the title-derived set is biased toward text,
and the UI set is only 12 questions), one run per variant, and answers judged by hand.
Production logs cannot answer the question, because OBS-002 logs a chunk count, not
which chunks were used.

## Pilot design

Question: does a description prompt anchored to the page produce better screenshot
chunks than the current one, and can the stored UUIDs be re-derived from the live page?

- Pages: `article/449` (partial UUID match seen earlier), `article/281` (contains the
  81k-character runaway), `article/441` (the License manager screenshot that hurt the
  library question).
- For each page: fetch the live HTML, extract `<img>` URLs, text before each image and
  alt text, recompute `uuid5(image_url)` and match it to the stored chunks.
- Re-describe each image with `gemini-3.6-flash`. The new prompt gives the page title,
  the text just before the image and the alt text, and asks for 60 to 150 words of plain
  text that starts with the page title and names the window or panel, the exact visible
  labels and values, and the action the screenshot illustrates. It forbids describing
  colours, pixel layout and decorative elements, and forbids enumerating long runs of
  similar items.
- Model settings: `thinking_level="low"`, `max_output_tokens=900`, `temperature=0.2`.
- Read-only: the viewer key was used and no object was changed.

## Results

### UUID re-derivation

| Page | Stored screenshot chunks matched by a live image URL | Notes |
|---|---|---|
| 449 | 3 of 4 | 1 stored chunk (873 characters) matches no live image: replaced or removed since March |
| 281 | 19 of 19 | |
| 441 | 7 of 7 | |

Site logos and one image (`confirm-deactivate.png`) have no stored chunk, as expected.
Re-deriving the mapping from a re-crawl works for most images but is not guaranteed.

### Old versus new descriptions (6 complete pairs)

| Page | Image | Old chars | New chars | New words | Distinct-word ratio |
|---|---|---|---|---|---|
| 449 | `file-ygMxdulzkt.png` | 1,249 | 497 | 72 | 0.72 |
| 449 | `file-CeswURJjOJ.png` | 3,193 | 678 | 105 | 0.73 |
| 449 | `file-5oWHaEqdXb.png` | 3,162 | 666 | 106 | 0.75 |
| 281 | `creating_a_sandwich_plate_001.png` | 3,139 | 626 | 99 | 0.74 |
| 281 | `creating_a_sandwich_plate_002.png` | 2,851 | 648 | 90 | 0.76 |
| 281 | `creating_a_sandwich_plate_004.png` | 2,770 | 702 | 103 | 0.79 |

The new descriptions are roughly 2.5 to 5 times shorter (4.3 times on average), contain no repetition and all ended
with a normal `STOP`. They keep exact labels and values that a question would use (for
example `LC040DB-E`, `Cetop counter pattern > Sandwich_Cetop03`,
`SD1E-A2_H2S5 [Argo-Hytos]`) and state which step of the page the image shows. The old
ones spend their length on layout and colour ("dark gray grid", "Local tab selected, gray
background").

### Examples

**Page:** Tutorial: Creating slip-in cartridge components  
**Image:** `file-ygMxdulzkt.png`  
**Text before the image on the page:** "his example we use Bosch Rexroth components. Go to the TAB online-> components-> Manufacturer-> BoschRexroth-> Control valves - Slip-In Cartridge-> Pressure control-> 2-way slip-in cartridges-> NS40 and select LC040DB-E."

*Old (1,249 characters, first 420 shown):*

> The screenshot displays a dark gray grid background with light gray lines, suggesting a technical drawing or CAD environment. In the center of the image, there is a white line drawing of an object. This object appears to be composed of several geometric shapes: * At the top, there's a rectangular shape with a zigzag pattern on its left side, resembling a spring or a heat exchanger. Inside this rectangle, to the right…

*New (72 words):*

> On the page "Tutorial: Creating slip-in cartridge components": The schematic canvas view shows the graphic symbol of a selected Bosch Rexroth 2-way slip-in cartridge valve (model LC040DB-E / size NS40). The symbol displays connection ports labeled C, B, and A, alongside the nominal size designation label NG40. The screenshot illustrates the placement and initial orientation of the slip-in cartridge component schematic symbol onto the drawing grid canvas during the component selection process.

---

**Page:** Tutorial: How to create a sandwich plate in Hydroman  
**Image:** `creating_a_sandwich_plate_002.png`  
**Text before the image on the page:** "he library go to Standards-> Cetop standards-> Cetop top pattern. Select the Sandwich top_Cetop03 and drag this into you schematic. Go to Cetop counter pattern and select Sandwich _Cetop03. Drag this into your schematic."

*Old (2,851 characters, first 420 shown):*

> This screenshot displays a software interface, likely a component or symbol library, with a hierarchical navigation pane on the left and a preview/selection pane on the right. **Top Bar (Window Controls):** * Standard Windows minimize, maximize/restore down, and close buttons are visible in the top right corner. **Main Navigation Tabs:** * Two prominent tabs are at the top: "Local" (currently selected, indicated by a…

*New (90 words):*

> On the page "Tutorial: How to create a sandwich plate in Hydroman": the image shows the library panel with tabs for Local and Online, alongside tab titles Components, Components+, and Symbols. Under the selected Components tab, the directory tree is navigated through Standards -> Cetop standards to the highlighted folder Cetop counter pattern. The right preview pane shows component choices including Sandwich top_Cetop03 [generic], Sandwich_Cetop02 [generic], and the selected item Sandwich_Cetop03 [generic]. The screenshot illustrates selecting the Sandwich_Cetop03 component from the Cetop counter pattern category to drag into the schematic.

---

**Page:** Tutorial: How to create a sandwich plate in Hydroman  
**Image:** `creating_a_sandwich_plate_004.png`  
**Text before the image on the page:** "schematic. In your schematic it will look like this. The upper component is your sandwich valve side and the lower component is your block side. Now insert your 2/2 solenoid valve. In this case a Argo Hytos SD1E-A2_H2S5."

*Old (2,770 characters, first 420 shown):*

> This screenshot displays a software interface for selecting and managing components, likely for engineering or design purposes. **Overall Layout:** The window has a standard title bar at the top with minimize, maximize/restore, and close buttons. Below the title bar, there are two main tabs: "Local" (currently selected, indicated by a grey background) and "Online" (unselected, with a light blue background). **Left Pa…

*New (103 words):*

> On the page "Tutorial: How to create a sandwich plate in Hydroman": The screenshot shows the component library panel in Hydroman, featuring the tabs Components, Components+, and Symbols, along with Local and Online toggle buttons. In the tree view under Manufacturer, the navigation path goes through Argo-Hytos, Control valves - On-off, Directional control, 22 - Normally closed, Solenoid operated, Poppet type - Direct, Cavity, down to 3/4-16 UNF. The preview area on the right displays the selected hydraulic valve symbol labeled SD1E-A2_H2S5 [Argo-Hytos]. This screenshot illustrates locating and selecting the Argo-Hytos SD1E-A2_H2S5 2/2 solenoid valve in the library to insert into the schematic.

### One flaw found: anchoring can import the wrong product name

In the `plate_004` description the new text says "the component library panel in
**Hydroman**". That panel is HydroSym's library; the neighbouring descriptions (`001`,
`002`) say HydroSym. The model took "Hydroman" from the page title ("…sandwich plate in
Hydroman"). The prompt should say: name the application only if its name is visible in the
screenshot.

### Model settings learned

- `max_tokens` is not a valid Gemini setting and raises a validation error; the field is
  `max_output_tokens`.
- `gemini-3.6-flash` spent about 2,100 reasoning tokens on a 60-word description by
  default. `thinking_level="low"` removes that cost. Reasoning tokens count against
  `max_output_tokens`, so the cap must leave room for them.

## Limits of the pilot

- **Incomplete.** After 8 successful calls the Gemini API returned `429 RESOURCE_EXHAUSTED`
  ("exceeded your current quota") for most remaining images. Some later calls succeeded,
  which suggests a rolling limit rather than a hard daily one, but this was not
  confirmed. The run was stopped by hand. Only 6 pairs (plus 2 site logos that are never
  indexed) are complete; none of the 9 images on page 441 were described.
- **Accuracy not verified.** The new descriptions were compared with the old ones and the
  page text. They were not checked image by image against the screenshots.
- **Retrieval benefit not re-measured.** The with/without comparison above used the
  existing descriptions. It has not been re-run with new ones.
- Shared key risk: if the local `GEMINI_API_KEY` is the Cloud Run secret, the pilot
  consumed quota that the live app also uses. No errors appeared in Cloud Run logs
  during the pilot; the keys were not compared.

## Decision

Use `gemini-3.6-flash` for the full re-description. It needs a key with enough quota to
describe about 380 images, sent sequentially with a pause between calls and a backoff on
`429`; a run of that size will take an hour or more.

## Proposed procedure for the full run

1. **Prompt changes**, from the pilot: page-anchored prompt; name the application only if
   visible; `thinking_level="low"`; `max_output_tokens` with headroom for reasoning; a
   repetition guard (reject a description with a low distinct-word ratio or over a length
   cap and retry once); skip decorative images such as logos.
2. **Provenance:** add an `image_url` property to the collection (no reindex needed; old
   objects read as null).
3. **Dry run for all pages**, no writes: write old and new descriptions to a local file
   and review a sample, especially for wrong product names.
4. **Snapshot** with `make snapshot-export` before any cloud write (the SYNC-005 rule).
5. **Apply by explicit UUID** with the admin key: replace the text, the vector (computed
   from `"<source_name>: <text>"`, as `add_chunks` does) and `image_url`. Do not rely on
   `batch.add_object` overwriting an existing UUID; that behaviour is untested.
6. **Unmatched chunks** (stored screenshot chunks with no live image, for example the
   873-character one on page 449) need an explicit decision per page: keep or delete.
7. **Re-measure:** re-run the 57-query retrieval measurement and the 12-question
   with/without comparison, and compare.

Acceptance: no screenshot chunk above a set length, no repetition loops, `image_url` set
on every screenshot chunk, and no regression on the 12 UI questions.

## Relation to Epic 11

This work belongs ahead of the diff logic in SYNC-001, because a sync cannot classify
screenshot chunks without `image_url`. The oversized-chunk cap also protects generation
and could be added independently as a limit on one chunk's share of the context budget.

## Reproducibility

The pilot, retrieval and comparison scripts were exploratory and are not in the
repository. The old and new text for the 6 pairs are shown above. Re-running requires the
hosted cluster URL, the viewer key, local Ollama with `nomic-embed-text`, and a Gemini key.
