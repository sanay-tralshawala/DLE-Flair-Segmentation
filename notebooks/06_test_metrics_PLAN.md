Please update `06_test_metrics.ipynb` to clean up the segmentation metrics reporting and make the notebook easier to interpret.

Context:
This notebook evaluates semantic segmentation models on the FLAIR-style merged land-cover classes. Since this is semantic segmentation, the primary metric should be mean IoU, with mean Dice and pixel accuracy as secondary metrics. The current notebook includes mAP, but we should drop it because it is not the most appropriate headline metric for this task. In this notebook, mAP is being computed from per-pixel class scores, not COCO-style instance segmentation mask AP, so it may confuse readers and overstate performance for classes that are ranked reasonably but rarely selected by argmax.

Main changes:
1. Remove mAP from the main evaluation outputs.
2. Remove mAP from summary tables and delta tables.
3. Remove or disable AP/mAP computation unless it is deeply entangled. If easier, leave the function unused but do not report it.
4. Use these metrics as the official notebook metrics:
   - Pixel Accuracy
   - Mean IoU / mIoU
   - Mean Dice
   - Per-class IoU
   - Per-class Dice
   - Per-class support
5. Add explanatory markdown cells that define the metrics and explain why mIoU is the primary metric.
6. Add a short interpretation section at the end summarizing the model comparison.

Please preserve the existing notebook flow as much as possible and avoid a large refactor.

Specific edits:

A. Add a markdown cell near the top after the notebook title:

## Evaluation Metrics

This notebook evaluates semantic segmentation models using overlap-based pixel metrics. The primary metric is **mean Intersection-over-Union (mIoU)** because each pixel is assigned exactly one semantic class, and the main goal is to measure spatial overlap between predicted and ground-truth regions.

Reported metrics:

- **Pixel Accuracy**: Fraction of valid pixels classified correctly. This is intuitive, but can be dominated by large/common classes.
- **Per-class IoU**: Intersection-over-union for each class: true overlap divided by the union of predicted and true pixels.
- **Mean IoU (mIoU)**: Average IoU across classes. This is the primary metric because it treats each class more equally than pixel accuracy.
- **Per-class Dice**: Dice/F1-style overlap for each class.
- **Mean Dice**: Average Dice score across classes. This is a secondary overlap metric that is often easier to interpret alongside IoU.
- **Support**: Number of ground-truth pixels for each class. This helps identify class imbalance.

We intentionally do **not** report mAP as a main metric. Mean Average Precision is more common for object detection and instance segmentation, where models predict separate object instances with confidence scores. For this semantic segmentation task, mIoU and Dice are more directly aligned with the final hard segmentation masks.

B. In the metric computation code:
- Remove `average_precision_score` imports if they are only used for AP/mAP.
- Remove `ap` and `mAP` from returned metric dictionaries.
- Remove AP from `class_rows`.
- Keep confusion matrix based calculations for IoU, Dice, pixel accuracy, and class support.

The resulting per-class row should look like:

{
    "class_id": class_id,
    "class_name": class_name,
    "support_pixels": support,
    "pred_pixels": pred_count,
    "iou": iou,
    "dice": dice,
}

The resulting model metrics should look like:

{
    "pixel_accuracy": pixel_accuracy,
    "mIoU": mean_iou,
    "mean_dice": mean_dice,
    "num_pixels": total_valid_pixels,
    "class_rows": class_rows,
}

C. Update the main summary table to include only:

Model | Input Channels | Pixel Acc | mIoU | Mean Dice

Do not include mAP.

D. Update the per-class tables to include only:

ID | Class | Support | Predicted | IoU | Dice

Do not include AP.

E. Update the 3-channel vs 5-channel comparison table to include only:

Model Family | 3ch mIoU | 5ch mIoU | Delta mIoU | 3ch Dice | 5ch Dice | Delta Dice | 3ch Acc | 5ch Acc | Delta Acc

Do not include Delta mAP.

F. Add a class support markdown/table section before or after the main summary:

## Class Support

Class support shows how many ground-truth pixels belong to each class. This matters because pixel accuracy can be dominated by large classes, while small classes may have poor IoU even if overall accuracy looks high.

If there is already enough information in the per-class tables, add a short support summary table using the first evaluated model’s `class_rows`, since support should be the same across models on the same test set.

Suggested columns:

ID | Class | Support Pixels | Support %

G. Add a best-model summary cell after the main comparison table:

## Best Model Summary

Automatically identify:
- Best model by mIoU
- Best model by Mean Dice
- Best model by Pixel Accuracy

Use mIoU as the primary ranking metric.

Suggested code:

best_by_miou = max(summary_rows, key=lambda row: row["mIoU"])
best_by_dice = max(summary_rows, key=lambda row: row["mean_dice"])
best_by_acc = max(summary_rows, key=lambda row: row["pixel_accuracy"])

display(Markdown(f"""
## Best Model Summary

- Best by **mIoU**: `{best_by_miou["model"]}` with `{best_by_miou["mIoU"]:.4f}`
- Best by **Mean Dice**: `{best_by_dice["model"]}` with `{best_by_dice["mean_dice"]:.4f}`
- Best by **Pixel Accuracy**: `{best_by_acc["model"]}` with `{best_by_acc["pixel_accuracy"]:.4f}`

Because this is a semantic segmentation task, **mIoU is treated as the primary metric**.
"""))

H. Add a final markdown interpretation section at the end. Use the actual computed results from the notebook, but frame the conclusion like this:

## Interpretation

The primary comparison should be based on **mIoU**, with Mean Dice and Pixel Accuracy used as secondary metrics. Pixel Accuracy is useful but can be misleading when classes are imbalanced, since large classes contribute many more pixels than rare classes.

The 3-channel vs 5-channel comparison should be interpreted per architecture rather than globally. If the extra IR/height channels improve ConvNeXt-based models but not ResNet34-UNet, say that the benefit of 5-channel input appears architecture-dependent.

Also call out the weakest per-class IoU results. If classes such as bare non-vegetated or herbaceous vegetation have low IoU, mention that minority or visually ambiguous classes remain the main limitation.

Avoid saying that DINOv3 is clearly superior unless the mIoU margin is large. If the top models are close, describe the result cautiously.

I. Standardize display names:
- Use `ConvNeXt-Tiny`, not `ConvNext-Tiny`
- Use `DINOv3-ConvNeXt-Tiny`, not `DINOv3-ConvNext-Tiny`

J. Optional but recommended:
Add a note near the config/model loading code explaining channel selection:

The 3-channel models should be evaluated on the same 3 channels they were trained on, and the 5-channel models should be evaluated on the same 5 channels they were trained on. If channel lists are currently inferred from `in_channels`, please make sure this is correct for the dataloader’s channel indexing convention.

Acceptance criteria:
- Notebook runs top to bottom.
- No main table reports mAP.
- No delta table reports mAP.
- Markdown clearly explains Pixel Accuracy, IoU, mIoU, Dice, and Support.
- mIoU is clearly described as the primary metric.
- Final interpretation is cautious and does not overclaim.