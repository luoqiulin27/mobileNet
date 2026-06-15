# Detection-First Development Guide

## 1. Project Goal

Build a lightweight indoor and outdoor obstacle detection system based on a MobileNet backbone.

Current main task:

- image in
- obstacle detections out
- each detection contains class, confidence, and bounding box

Current non-goals:

- reminder generation
- route planning
- `safe / left / right / stop` image classification
- full autonomous navigation

## 2. Core Constraint

All future development must align with this path:

1. dataset preparation
2. label mapping
3. lightweight detection baseline
4. evaluation and error analysis
5. export and mobile deployment preparation

Do not reintroduce a classification-first pipeline as the default direction.

## 3. Data Usage Policy

Available datasets:

- outdoor: `SANPO-Real-Labeled-Full`
- indoor: `SUNRGBD`

Rules:

1. Prefer detection supervision when available.
2. If segmentation labels are used, convert them into detection annotations explicitly and document the conversion rule.
3. Indoor and outdoor data should be validated separately before any merged training.
4. Keep a unified class mapping file under the active detection project.

## 4. Baseline Definition

The default baseline should be:

- MobileNet backbone
- lightweight detection head
- trainable on current workspace
- exportable to ONNX later

Recommended implementation order:

1. outdoor detection baseline
2. indoor detection baseline
3. unified label-space experiment
4. deployment-oriented optimization

## 5. What Counts As Relevant Code

Relevant:

- detection datasets
- detection models
- detection losses
- detection training scripts
- detection evaluation scripts
- detection demo code
- export utilities

Not relevant to the current mainline:

- global image classification into driving or reminder labels
- rule-based reminder text generation
- decision engines that assume one label per image
- audio output modules

## 6. Task Template For Future Work

Every new coding task should be framed like this:

- goal
- input
- output
- files to touch
- constraints
- acceptance criteria

## 7. Acceptance Standard

A change is acceptable only if it improves at least one of:

- annotation quality
- detection training stability
- detection accuracy
- inference speed
- export readiness
- maintainability of the detection codebase

## 8. Current Active Directory

Use this as the default working project:

- `D:\project\mobileNet\blind-assist-detection`

Anything outside it should be treated as legacy unless explicitly retained.
