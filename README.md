# Generating Sinusoidal & Helical data via Denoising Diffusion Probabilistic Models
---

Implemented DDPM  for 3D Helical & Sinusoidal dataset.
- In this project, we built a Feed Foward Neural Network based Denoiser, which modelled the above-mentioned datasets. 
- The models were built from scratch, and we also implemented various noise schedulers to perform a comparative study and empirically decide which noise scheduler works better.
- We implemented Linear, Quadratic, Root & Cosine schedulers and studied which one worked the best.
- We used the metrics such as Earth Mover's Distance, Chamfer Distance & Negative log-likelihood to make the comparisons.
- We successfully built a model which could mimic the data well.

---
- `best_helix` & `best_sin` folders contain a graphical representation of the samples generated from the corresponding distributions.
- `main_folder` contains the best-performing models and other comparisons.
---
