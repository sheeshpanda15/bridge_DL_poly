# Theory presentation figure prompts

Generated with the built-in image_gen tool. Each image is a conceptual scientific illustration, not an experimental plot.

## 01_model_to_shape_space.png

```text
Use case: scientific-educational.
Asset type: a standalone landscape 16:9 academic presentation illustration, ideally 2560 x 1440 or higher. This is one image, not a slide mockup photographed in a room.
Style: polished scientific figure with elegant translucent 3D mathematical surfaces, fine wireframe lines, simple vector-like diagram elements and beautifully typeset short labels. Pure white background, generous whitespace, navy typography, consistent teal (#087f8c) for NN/reference, warm orange (#ed8d40) for polynomial/comparison, violet (#7759b5) for geometric connections. No decorative unrelated objects. All labels in English, crisp and legible at presentation scale. Balanced composition with 6% safe margins. No watermark, no brand logo. Diagrams are conceptual, never pretend to be experimental data.
Primary request: Explain this actual research idea: compare a neural network and a polynomial regression model by the shapes of their sampled response surfaces; each whole point cloud represents one shape-space point. Build a clear three-stage left-to-right story with subtle arrows.
Exact large title: "From response surfaces to shape space".
LEFT, label "01  Model responses": two small stacked 3D Cartesian plots with the same x1,x2 domain. Top a smooth teal mildly nonlinear response surface labelled "Neural network"; bottom a similar orange polynomial response surface labelled "Polynomial model". Axes labels only "x1", "x2", "prediction"; no numerical ticks. Show sampled points on both surfaces.
MIDDLE, label "02  Matched point clouds": two delicate 3D point-cloud representations, teal and orange, with corresponding sampled input locations, the response coordinate varies. Underneath use typeset formula "G_f = [(x_i, f(x_i))]" and short caption "Same input points". Next to the outgoing arrow put two short lines "Center & normalize" and "Align".
RIGHT, label "03  Shape representation": a translucent pale blue curved manifold patch floating in white space with fine grid and two individual large dots on it, one teal labelled "[G_NN]", one orange labelled "[G_PR]". Each dot must be just a single point, not a whole cloud. A subtle violet curved connecting line indicates conceptual separation. Nearby annotation "One model cloud = one shape".
Exact bottom caption: "Compare geometric structure to connect neural networks and polynomial models."
Tiny but legible footer: "Conceptual schematic of shape space".
Constraints: Each 3D response surface on left is a graph over input coordinates; the curved patch on right is an abstract representation space and must not be confused with a response surface. No numerical result, no performance claim, no formulas for a computed geodesic.
```

## 02_preshape_geodesic.png

```text
Use case: scientific-educational.
Asset type: a standalone landscape 16:9 academic presentation illustration, ideally 2560 x 1440 or higher. This is one image, not a slide mockup photographed in a room.
Style: polished scientific figure with elegant translucent 3D mathematical surfaces, fine wireframe lines, simple vector-like diagram elements and beautifully typeset short labels. Pure white background, generous whitespace, navy typography, consistent teal (#087f8c) for NN/reference, warm orange (#ed8d40) for polynomial/comparison, violet (#7759b5) for geometric connections. No decorative unrelated objects. All labels in English, crisp and legible at presentation scale. Balanced composition with 6% safe margins. No watermark, no brand logo. Diagrams are conceptual, never pretend to be experimental data.
Primary request: A beautiful scientifically restrained illustration of spherical Riemannian geometry as the motivation for comparing centered, normalized model point clouds. Crucial: the current project computes Procrustes disparity, so never label this illustration as its computed distance.
Exact title: "The geometric idea: distance along shape space".
Small subtitle: "A schematic view of aligned pre-shape representatives".
Main illustration occupying left 65%: a translucent pale teal unit sphere with latitude/longitude-like fine grid, seen in 3D perspective on white, with two large points A (teal) and B (orange) on the upper front hemisphere, separated by roughly 65 degrees. Connect these SAME two endpoints by a prominent violet short great-circle arc that stays on the sphere surface, label it "Spherical geodesic". Also connect SAME endpoints by a thin dashed gray straight chord inside the sphere, label "Ambient chord". The curved arc and straight chord must be visibly different. Do not add another path, tube, arrowhead on surface, or tangent plane. Make sphere subtle and readable.
Right 30%: clean formula stack with two compact labels.
"Normalize each cloud"
"A = (G - mean(G)) / ||G - mean(G)||_F"
"On the unit sphere"
"||A||_F = ||B||_F = 1"
"Arc-length intuition"
"d_sph(A, B) = arccos(<A, B>_F)"
Render as professionally typeset mathematical notation, with appropriate subscripts and angle brackets, not literal source markup.
Below formulas exact two short sentences: "Each point represents an entire configuration." "Alignment removes orientation differences."
Bottom full-width pale very light violet strip, clearly legible: "Theory illustration. The project currently uses Procrustes disparity."
Constraints: sphere represents a schematic slice of a high-dimensional pre-shape sphere, not a claim that the entire Kendall space is a 2-sphere. Do not print 'Kendall distance equals Procrustes disparity'. All geometric labels must point to correct arc/chord. No fabricated data.
```

## 03_procrustes_distance.png

```text
Use case: scientific-educational.
Asset type: a standalone landscape 16:9 academic presentation illustration, ideally 2560 x 1440 or higher. This is one image, not a slide mockup photographed in a room.
Style: polished scientific figure with elegant translucent 3D mathematical surfaces, fine wireframe lines, simple vector-like diagram elements and beautifully typeset short labels. Pure white background, generous whitespace, navy typography, consistent teal (#087f8c) for NN/reference, warm orange (#ed8d40) for polynomial/comparison, violet (#7759b5) for geometric connections. No decorative unrelated objects. All labels in English, crisp and legible at presentation scale. Balanced composition with 6% safe margins. No watermark, no brand logo. Diagrams are conceptual, never pretend to be experimental data.
Primary request: Explain the exact practical shape discrepancy used in this project with matched point configurations. White academic diagram, 4 equal spacious panels arranged in one horizontal row under a title, teal point set A and orange point set B throughout. Four sets of connected landmarks should be easy to see, using 8 correspondingly ordered points forming a softly bent irregular open chain in 2D as a schematic of higher-dimensional response clouds, not arbitrary unordered dots. No numbering on individual landmarks.
Exact large title: "Computing the Procrustes shape discrepancy".
Panel 1 label "1  Center": show the original two comparable configurations displaced from each other with faint outlines, then centered versions sharing a small gray cross at the origin. Tiny caption "Remove translation".
Panel 2 label "2  Normalize": show two centered configurations with matched overall scale in a faint circular guide; the orange one is still visibly rotated relative to teal. Tiny caption "Unit Frobenius norm".
Panel 3 label "3  Align": show orange configuration rotating toward teal with a clean curved arrow, almost aligned yet retaining small shape differences. Tiny caption split across 2 lines "Orthogonal transform" and "+ optimal scale".
Panel 4 label "4  Measure": show teal and transformed orange configurations nearly overlapping with several short thin purple residual segments connecting corresponding points. Tiny caption "Sum squared residuals".
Connect panels using simple light gray horizontal arrows.
Under the panels display one prominent correct formula in professional math typography:
"D(A,B) = min_{s >= 0, R^T R = I} ||A - sBR||_F^2"
Under formula: "A and B are centered, unit-norm configurations."
Bottom left legend dots "Reference" teal, "Comparison" orange, "Residual" violet.
Bottom right two short clear notes: "Reflections are allowed." and "SciPy Procrustes disparity; not geodesic arc length."
Constraints: orthogonal alignment permits reflection, and a further isotropic scale s is optimized after each input is normalized. Do not state 'rotation only', do not take a square root of the output, do not call disparity a proven Riemannian metric. No added statistics.
```

## 04_distance_guided_sampling.png

```text
Use case: scientific-educational.
Asset type: a standalone landscape 16:9 academic presentation illustration, ideally 2560 x 1440 or higher. This is one image, not a slide mockup photographed in a room.
Style: polished scientific figure with elegant translucent 3D mathematical surfaces, fine wireframe lines, simple vector-like diagram elements and beautifully typeset short labels. Pure white background, generous whitespace, navy typography, consistent teal (#087f8c) for NN/reference, warm orange (#ed8d40) for polynomial/comparison, violet (#7759b5) for geometric connections. No decorative unrelated objects. All labels in English, crisp and legible at presentation scale. Balanced composition with 6% safe margins. No watermark, no brand logo. Diagrams are conceptual, never pretend to be experimental data.
Primary request: Clearly illustrate the actual application of this project: pilot NN-versus-polynomial shape discrepancies determine a fixed mixture of polynomial D-optimal samples and random samples for subsequent training batches. Design a wide elegant three-stage conceptual research diagram.
Exact large title: "From model distance to sampling design".
LEFT section label "01  Compare pilot models": a small pale input-data cloud leads to a minimal neural-network icon and two small smooth polynomial-curve icons labelled "NN", "FullPR", "Taylor-PR". Pairwise comparison connectors feed a small violet pill labelled "Shape discrepancy". Under it typeset cleanly "d = (d_FPR + d_TYPR) / 2". Small footnote here "Grand benchmark: mean combination".
MIDDLE section label "02  Set the sampling weight": a real mathematically correct schematic Cartesian plot, horizontal axis "Normalized discrepancy" ranging 0 left to 1 right, vertical axis "D-optimal fraction" with values 0.95 at upper left and 0.30 at lower right. The only plotted line is a straight descending teal line from (0, 0.95) to (1, 0.30), not a curve. Label near its upper-left end "More D-optimal", near lower-right end "More random". Under plot put formula "w = 0.30 + 0.65 (1 - d_norm)".
RIGHT section label "03  Build each new batch": a large horizontal two-color segmented bar mostly teal with a smaller orange part, labels above or inside "D-optimal" and "Random", and "w" and "1 - w" directly below their corresponding segments. Underneath a square point-cloud icon with a mixture of teal informative samples and scattered orange random samples feeding via an arrow into a small NN retraining icon labelled "Retrain NN". One simple loop arrow runs from retraining back to the batch bar, entirely contained in this right section.
Connect left to middle and middle to right with subtle gray arrows.
Bottom caption: "Closer polynomial geometry increases the share of design-based samples."
Legible footer: "Pilot weight stays fixed across rounds; the selected sample set grows."
Constraints: Do not draw a feedback arrow back to the pilot-distance stage; project does not recompute weights each round. Do not suggest every nearer geometry guarantees better performance. The plot is a policy mapping, not an empirical finding. D-optimal teal and random orange. The 0.30 and 0.95 values refer to documented grand benchmark settings. Keep annotations sparse, no dense flowchart.
```

## 02_preshape_geodesic_no_title.png

User-requested removal of the masked title and subtitle. Original preserved.

```text
Use case: precise-object-edit.
Edit target: the attached scientific presentation image.
Request: remove exactly the region highlighted by the user's white mask on a black background. The highlighted area is the top title and subtitle band, approximately x=195..1570 and y=0..124 in this 1672x941 image. Specifically remove all lettering of "The geometric idea: distance along shape space" and "A schematic view of aligned pre-shape representatives". Seamlessly fill that cleared region with the same clean white background as surrounding image.
Preserve invariants: maintain the exact original canvas size and framing. Do not crop, recenter, enlarge, or move any other content. Preserve the sphere, A/B markers, paths, right-hand equations, every remaining label, the bottom violet note, colors, line styles and layout. No replacement title or subtitle, no new content. Everything below y=125 should be unchanged. This is a local title-removal edit only.
```

