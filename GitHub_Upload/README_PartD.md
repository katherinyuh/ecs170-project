# Stage 4 Part D experiments

This contribution adds parameter experiments for both text generation and classification. The package contains 21 completed full runs: 18 generation runs and three classification GRU runs. The team models and loaders are reused without editing teammates' model files.

## Upload using the GitHub website

In the downloaded archive, open `GitHub_Upload`. Its contents belong at the root of `katherinyuh/ecs170-project`, alongside the existing `local_code` and `script` folders.

1. Open https://github.com/katherinyuh/ecs170-project while signed in.
2. Stay at the repository root; do not open `local_code` first.
3. Choose **Add file > Upload files**.
4. Open `GitHub_Upload` on your computer. Drag its contents onto the upload page. Do not drag the outer `GitHub_Upload` folder itself.
5. Check that the paths include `script/stage_4_script/run_part_d.py` and `result/stage_4_result/part_d/`. The existing teammates' files are not replaced.
6. Use the commit message `Add Stage 4 Part D parameter experiments and results`, then commit using your team's branch workflow.

If GitHub offers a proposed change instead of a direct commit, the repository may require a branch, pull request, or additional access. Follow the team's review workflow.

The notebook is an `.ipynb` file stored on GitHub. It can also be opened in Colab to rerun the experiments. The measured results are already included and do not need to be rerun before uploading.

The compact main report is also included in `GitHub_Upload`. Supplied Part B references and the detailed generation supplement are in `Report_and_References`. The main report contribution is concise so the team can stay within the five-page total.

## Completed experiments

Generation uses the same 1,622-joke dataset and 30,303 next-token windows as Part C. Every setting runs RNN, LSTM and GRU for 15 epochs.

| Variation | Single changed setting |
| --- | --- |
| baseline | Embedding 100, hidden 128, one layer, context 3, learning rate 0.001 |
| embedding50 | Embedding dimension 100 to 50 |
| hidden256 | Hidden state 128 to 256 |
| layers2 | One recurrent layer to two |
| sequence | Context length 3 to 5 tokens |
| lr0005 | Learning rate 0.001 to 0.0005 |

Classification focuses on GRU, which had the highest supplied Part B baseline accuracy. Three runs compare baseline hidden size 128 and learning rate 0.001 with hidden size 256 or learning rate 0.0005. They all use fine-tuned GloVe 100d, a 200-token review limit, one recurrent layer and three epochs.

Classification reserves a fixed stratified 2,500-review validation set from the original training reviews, leaving 22,500 for fitting. It scores the original 25,000-review test set once after the final epoch. Compare variants against the new Part D baseline. Part B used all 25,000 training reviews, so its supplied scores are a separate reference rather than the controlled baseline for these ablations.

Seed 2, Adam, batch size 64 and gradient clipping 5 stay fixed. Embedding/output dropout is 0.2 for generation or 0.3 for classification. Recurrent dropout stays at zero at both depths to isolate depth. These are single-seed comparisons.

## Run locally

Copy the contents of `GitHub_Upload` into the existing project, retaining the folder paths. With the instructor's datasets in `data/stage_4_data/`, run from the project root:

```bash
python script/stage_4_script/run_part_d.py --task both --quick
python script/stage_4_script/run_part_d.py --task generation
python script/stage_4_script/run_part_d.py --task classification --cells gru --variants baseline hidden256 lr0005 --epochs 3 --glove /path/to/glove_subset_100d.txt
```

The classification file can be the matching 100d subset or the original 100d GloVe archive. The Colab notebook downloads the instructor-linked archive and exports the 19,345 matching vocabulary vectors when needed. To avoid that larger download, upload the team's `glove_subset_100d.txt` into Colab first. The 50d file cannot reproduce a 100d baseline.

Quick runs train for one epoch on 256 examples and are installation checks only. They are stored separately from report results. Generation learns its embeddings from scratch.

Completed runs with matching settings are skipped. A settings/data mismatch is rejected, so use a different `--out` directory when changing epochs, embedding sources or datasets. Add `--save-weights` to retain model weights and vocabulary. The packaged outputs have no model weight files.

For a supported wider sweep, `--plan --task both` prints the available settings. A pretrained embedding-size comparison needs corresponding GloVe dimensions; the runner refuses the `embedding50` comparison with a fixed 100d file.

## Results and interpretation

`result/stage_4_result/part_d/` contains the combined 21-row CSV, histories, comparison curves and one folder per model/variation. Generation saves ten prompt outputs per model, with training-overlap checks. Classification saves test predictions and accuracy, macro precision, recall and F1.

Generation `train_eval_acc` measures next-token accuracy on training windows with dropout off. It is not held-out test accuracy or story quality. Training-curve loss and accuracy use dropout with weights changing during the epoch. A copied-span count includes punctuation tokens. A full-output match means the entire generated output occurs as a contiguous substring in a training joke; it need not reproduce the whole joke.

For classification, per-epoch curves use validation examples. The reused model internally prints `TestAcc` for that supplied validation partition; the CSV's final `accuracy` is the score on the original test set. Validation accuracy is stored separately.

Part D's measured runs used CPU. Timing was collected on shared resources and is descriptive, not a CPU/GPU benchmark. The Colab notebook supports CUDA but has not been executed on a GPU in this workspace. Numerical results can vary across hardware and PyTorch versions.

## Supplied Part B reference

The download's `Report_and_References/references/part_b/` folder preserves the supplied results and plots. Final classification test accuracies are RNN 0.79156, LSTM 0.87184 and GRU 0.89572. Use final-epoch CSV values, not each model's maximum test-curve value. Those reference runs remain separate from the new Part D measurements.

## Terminal alternative

The website workflow above is sufficient. If using a local Git checkout instead:

```bash
git add script/stage_4_script/run_part_d.py Stage4_PartD_Colab.ipynb README_PartD.md PartD_Report_Sections.txt Stage4_PartD_Report_Sections.docx
git add -f result/stage_4_result/part_d
git commit -m "Add Stage 4 Part D parameter experiments and results"
git push
```

The explicit `-f` is needed only for local Git because the team ignores `result/`. Do not add datasets, GloVe archives, virtual environments or temporary smoke checks.
