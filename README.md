# Multimodal Artist Similarity

Combining audio, images, and captions to measure how similar two musical artists are — and testing whether audio actually helps.

## The Finding

Adding audio as a third modality does **not** improve artist similarity assessment. The best model uses only Instagram images and captions (**98.2%** accuracy); adding audio consistently lowers it (**97.2%**). The reason is redundancy: audio similarity is largely already captured by image + caption similarity (**r = 0.68** over 10,000 random artist pairs). To help, a modality must be good *and* different — audio is good, but not different.

## The Four Experiments

All experiments: 2,895 artists (the intersection with all three modalities), 3 random seeds, same 1D-CNN Siamese network with triplet loss (cosine distance), tested on unseen data.

| # | Experiment | Result |
|---|---|---|
| 1 | **Modality ablation** — all 7 combinations of image / caption / audio | Image + caption best (98.21% ± 0.12); every audio configuration ranks below its audio-free setup |
| 2 | **Caption encoder** — English vs. multilingual | No real difference; the finding holds under both |
| 3 | **Ground truth source** — AllMusic vs. Spotify vs. union | Audio never helps, no matter the source |
| 4 | **Hyperparameter tuning** — margin, dropout, batch size (one-factor-at-a-time) | Best tuned 3-modality model (97.90%) still below image + caption (98.21%) |

## Repository Structure

```
experiments/   Training scripts for the four experiments + SLURM job files
analysis/      Analysis notebook (t-SNE, correlation, per-genre/nationality effects)
figures/       Figures reported in the thesis and presentation
data/          Not included — see data/README.md
```

## Credits

This work builds on two prior theses at Politecnico di Milano:

- **Davide Lista** — audio-based similarity model and data collection
- **Giacomo Sansoni** — Instagram-based (image + caption) similarity model

Research by **Ali Rahmatpour**, Politecnico di Torino.
Supervisors: **Prof. Cristina Rottondi** and **Prof. Massimiliano Zanoni**.

## Contact

- LinkedIn: [linkedin.com/in/ebteda](https://www.linkedin.com/in/ebteda/)
