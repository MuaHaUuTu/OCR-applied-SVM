OCR-applied-SVM/
│
├── app.py # Main script for running OCR + SVM inference

├── requirements.txt # Required Python packages

├── charset_official_ver3.txt# Character set used during training/inference

├── svtr_config.yml # Configuration file for SVTR model (PaddleOCR backbone)

├── weight_svtr.pdparams # Pretrained weights for PaddleOCR model

│
├── train/ # Folder with training images

│ └── *.jpg / *.png # Training images

├── train_gt_fold1.txt # Annotation file (ground-truth) for training

│
├── valid/ # Folder with validation images

│ └── *.jpg / *.png # Validation images

├── valid_gt_fold1.txt # Ground-truth annotations for validation

│
└── svm_model.pkl # Saved scikit-learn SVM model (after training)
