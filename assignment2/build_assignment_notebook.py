from textwrap import dedent

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """
        # Assignment 2: Fashion-MNIST Baselines vs. a Simple CNN

        **Research question:** How much does a simple convolutional neural network (CNN) improve Fashion-MNIST image classification performance relative to classical machine learning baselines?

        This notebook is now organized to match the ten-part rubric. It is intentionally more explicit than a compact research notebook: every major step is named, explained, and tied to the assignment requirements.

        ## Important assignment-fit note

        The rubric says the dataset should come from **your own digital archive**. Fashion-MNIST is a public benchmark dataset, so this notebook should be treated in one of two ways:

        1. **As a modeling prototype/template** that you adapt to your own image archive later.
        2. **As a final submission only if your instructor explicitly allows a public dataset.**

        I am making that limitation explicit rather than pretending this dataset satisfies the rubric automatically.

        If you later swap in your own archive of labeled images, most of the notebook can stay the same:

        - Section 2 would change from `torchvision.datasets.FashionMNIST(...)` to loading your own files from folders.
        - Section 1 would need to describe where those files came from in your own life.
        - The later preprocessing, modeling, and evaluation sections would still be valid with only minor changes.
        """
    )
)

cells.append(
    md(
        """
        ## 1. Data Explanation

        This project uses **Fashion-MNIST**, a dataset of grayscale clothing images introduced by Xiao, Rasul, and Vollgraf (2017) as a more challenging replacement for MNIST digits. The dataset contains:

        - `60,000` training images
        - `10,000` test images
        - `10` clothing categories
        - one `28 x 28` grayscale image per item

        The ten classes are:

        1. T-shirt/top
        2. Trouser
        3. Pullover
        4. Dress
        5. Coat
        6. Sandal
        7. Shirt
        8. Sneaker
        9. Bag
        10. Ankle boot

        ### Why this dataset was chosen

        I chose this dataset because my first assignment suffered from weak predictive signal. Fashion-MNIST provides a much cleaner supervised-learning setting where:

        - the labels are already defined,
        - the class balance is uniform,
        - the visual differences between some classes are obvious and others are subtle,
        - and both classical models and deep-learning models can be compared meaningfully.

        ### How the data was obtained

        In this notebook the data is obtained programmatically through the `torchvision` dataset interface, which downloads the official Fashion-MNIST files and exposes them as Python objects. That means this notebook is reproducible, but it also means the sample is **not** drawn from my personal digital archive. If this notebook is adapted for final submission under the strict rubric, that mismatch should be resolved by replacing Fashion-MNIST with a personal archive of labeled images.

        ### How the sample is formed

        To keep the notebook fast enough to rerun on a normal laptop CPU, I use:

        - a stratified working training sample of `10,000` images,
        - a stratified validation sample of `2,000` images,
        - and the official `10,000` image test split as the final out-of-sample evaluation set.

        Stratification preserves the original class balance, so the reduced sample still represents the full dataset fairly.
        """
    )
)

cells.append(
    code(
        """
        import os
        import copy
        import random
        import time
        import warnings
        from pathlib import Path

        # Matplotlib writes a config directory on first import. Creating a local
        # writable directory avoids permission warnings inside the notebook runtime.
        mpl_dir = Path.cwd() / ".matplotlib"
        mpl_dir.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        import seaborn as sns
        from IPython.display import Markdown, display

        from sklearn.decomposition import PCA
        from sklearn.dummy import DummyClassifier
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision.datasets import FashionMNIST

        sns.set_theme(style="whitegrid", context="talk")

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        CLASS_NAMES = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

        WORKING_TRAIN_SIZE = 10_000
        WORKING_VAL_SIZE = 2_000
        BATCH_SIZE = 128
        EPOCHS = 12
        PATIENCE = 3

        if torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
        elif torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cpu")

        print(f"Using device: {DEVICE}")
        """
    )
)

cells.append(
    md(
        """
        ## 2. Converting the Data to Python-Readable Format

        The raw Fashion-MNIST files are stored in IDX format. Instead of parsing those files manually, this notebook uses `torchvision.datasets.FashionMNIST`, which:

        - downloads the official files,
        - verifies and reads them,
        - exposes the pixel grids as tensor-backed arrays,
        - and makes the labels available alongside the images.

        In other words, this section is the notebook's "data ingestion" layer. It converts external files into Python data structures that the rest of the analysis can use.

        The code below:

        - downloads or loads the dataset,
        - converts images to NumPy arrays scaled to `[0, 1]`,
        - creates a stratified train/validation split from the official training data,
        - and prepares both image-shaped arrays and flattened arrays for later models.
        """
    )
)

cells.append(
    code(
        """
        data_root = Path.cwd() / "data"

        # Load the official Fashion-MNIST training and test datasets.
        train_full = FashionMNIST(root=data_root, train=True, download=True)
        test_full = FashionMNIST(root=data_root, train=False, download=True)

        # Convert the image data into NumPy arrays and rescale pixels from
        # integer values in [0, 255] to floating point values in [0, 1].
        X_full = train_full.data.numpy().astype("float32") / 255.0
        y_full = train_full.targets.numpy()
        X_test = test_full.data.numpy().astype("float32") / 255.0
        y_test = test_full.targets.numpy()

        # Create a manageable working subset from the official training data.
        working_indices, _ = train_test_split(
            np.arange(len(y_full)),
            train_size=WORKING_TRAIN_SIZE + WORKING_VAL_SIZE,
            stratify=y_full,
            random_state=SEED,
        )

        # Split that subset into training and validation partitions.
        train_idx, val_idx = train_test_split(
            working_indices,
            train_size=WORKING_TRAIN_SIZE,
            stratify=y_full[working_indices],
            random_state=SEED,
        )

        X_train = X_full[train_idx]
        y_train = y_full[train_idx]
        X_val = X_full[val_idx]
        y_val = y_full[val_idx]

        # Classical models in scikit-learn expect tabular feature vectors, so
        # we also keep flattened versions of each image.
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_val_flat = X_val.reshape(len(X_val), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)

        split_summary = pd.DataFrame(
            [
                {"Split": "Train", "Samples": len(X_train), "Image shape": X_train.shape[1:]},
                {"Split": "Validation", "Samples": len(X_val), "Image shape": X_val.shape[1:]},
                {"Split": "Test", "Samples": len(X_test), "Image shape": X_test.shape[1:]},
            ]
        )
        display(split_summary)
        """
    )
)

cells.append(
    md(
        """
        At this point the data exists in two parallel forms:

        - **Image tensors** with shape `(n, 28, 28)` for the CNN.
        - **Flattened feature vectors** with shape `(n, 784)` for classical scikit-learn models.

        That distinction is important. Flattened vectors ignore spatial arrangement, while CNNs preserve local neighborhoods of pixels. Much of the later comparison is really about whether preserving that structure matters.
        """
    )
)

cells.append(
    md(
        """
        ## 3. Cleaning, Pre-Processing, Feature Engineering, and Exploratory Data Analysis

        This dataset is already much cleaner than real-world personal archives. There are no missing labels, no duplicate columns, and no mixed file formats. Even so, some preprocessing is still required.

        ### Cleaning and preprocessing choices

        - **Pixel scaling:** raw pixel values are scaled from `0-255` to `0-1`.
        - **Flattening for classical models:** logistic regression and Naive Bayes need tabular vectors rather than 2D images.
        - **PCA for dimensionality reduction:** principal component analysis is used as a feature-engineering step before one of the logistic-regression models.

        ### Why EDA matters here

        Before fitting any model, it is useful to verify:

        - whether the classes are balanced,
        - what the images actually look like,
        - whether some classes look visually similar,
        - and whether the basic summary statistics suggest any obvious issues.
        """
    )
)

cells.append(
    code(
        """
        # Show one example image from each class so the reader can see what the
        # prediction task looks like.
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))

        for class_id, ax in enumerate(axes.flat):
            example_idx = np.where(y_train == class_id)[0][0]
            ax.imshow(X_train[example_idx], cmap="gray")
            ax.set_title(CLASS_NAMES[class_id], fontsize=12)
            ax.axis("off")

        plt.suptitle("One example image from each Fashion-MNIST class", y=1.03)
        plt.tight_layout()
        plt.show()

        class_balance = pd.DataFrame(
            {
                "Class": CLASS_NAMES,
                "Train": np.bincount(y_train, minlength=len(CLASS_NAMES)),
                "Validation": np.bincount(y_val, minlength=len(CLASS_NAMES)),
                "Test": np.bincount(y_test, minlength=len(CLASS_NAMES)),
            }
        )
        display(class_balance)

        ax = class_balance.set_index("Class").plot(kind="bar", figsize=(14, 6))
        ax.set_ylabel("Number of images")
        ax.set_title("Class balance across data splits")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        # Compute a few basic descriptive statistics on pixel intensities.
        descriptive_stats = pd.DataFrame(
            [
                {
                    "Split": "Train",
                    "Mean pixel intensity": float(X_train.mean()),
                    "Std. dev.": float(X_train.std()),
                    "Min": float(X_train.min()),
                    "Max": float(X_train.max()),
                },
                {
                    "Split": "Validation",
                    "Mean pixel intensity": float(X_val.mean()),
                    "Std. dev.": float(X_val.std()),
                    "Min": float(X_val.min()),
                    "Max": float(X_val.max()),
                },
                {
                    "Split": "Test",
                    "Mean pixel intensity": float(X_test.mean()),
                    "Std. dev.": float(X_test.std()),
                    "Min": float(X_test.min()),
                    "Max": float(X_test.max()),
                },
            ]
        )
        display(descriptive_stats)
        """
    )
)

cells.append(
    code(
        '''
        display(
            Markdown(
                f"""
                ### EDA Takeaways

                - The sample is **perfectly class-balanced** because stratified sampling preserved the original label distribution.
                - Pixel intensities are already on a common scale because values were normalized to the interval `[0, 1]`.
                - Some classes are likely to be easy to distinguish visually, such as **bags** and **ankle boots**.
                - Other classes are likely to be harder because they have overlapping silhouettes, such as **shirts**, **pullovers**, and **coats**.

                These observations matter because they foreshadow what the confusion matrices later reveal: the hard errors are not random, but concentrated among visually similar clothing categories.
                """
            )
        )
        '''
    )
)

cells.append(
    md(
        """
        ## 4. Analysis Plan and Data Splits

        The task in this notebook is **multiclass classification**. Each image belongs to exactly one of ten clothing categories, so the model's job is to predict the correct class label.

        ### Why classification?

        - The target variable is categorical rather than numerical, so regression is inappropriate.
        - The labels are known in advance, so this is supervised learning rather than clustering.
        - The assignment question is naturally phrased as "Which type of clothing is shown in this image?"

        ### Data split strategy

        I use three splits:

        - **Training set:** used to fit model parameters.
        - **Validation set:** used to compare models and tune CNN hyperparameters.
        - **Test set:** used once at the end for out-of-sample evaluation.

        This is important because evaluating on the same data used for training would overstate performance.
        """
    )
)

cells.append(
    code(
        '''
        display(
            Markdown(
                f"""
                ### Split Summary

                - Training images: **{len(X_train):,}**
                - Validation images: **{len(X_val):,}**
                - Test images: **{len(X_test):,}**
                - Feature dimension for classical models after flattening: **{X_train_flat.shape[1]}**

                The training and validation sets come from the official Fashion-MNIST training split, while the test set remains untouched until the final evaluation stage.
                """
            )
        )
        '''
    )
)

cells.append(
    md(
        """
        ## 5. Model Selection and Mathematical Background

        I compare several models rather than jumping directly to a CNN. That makes the notebook more informative because it shows what is gained by the more advanced architecture.

        ### Models included

        1. **Majority-class baseline**
        2. **Gaussian Naive Bayes**
        3. **Logistic regression**
        4. **PCA + logistic regression**
        5. **A small CNN**

        ### 5.1 Majority-class baseline

        This model always predicts the most common class. It is intentionally weak, but it establishes a floor. If a sophisticated model cannot beat it, the project has failed.

        ### 5.2 Gaussian Naive Bayes

        Naive Bayes assumes conditional independence of features given the class:

        $$
        p(y \\mid x_1, \\dots, x_d) \\propto p(y) \\prod_{j=1}^{d} p(x_j \\mid y)
        $$

        In the Gaussian version, each feature is modeled as normally distributed within a class. This assumption is clearly unrealistic for images, but it still provides a fast probabilistic baseline.

        ### 5.3 Multinomial logistic regression

        Logistic regression is a linear classifier. For multiclass prediction, it uses the softmax function:

        $$
        p(y = k \\mid x) = \\frac{\\exp(w_k^T x + b_k)}{\\sum_{j=1}^{K} \\exp(w_j^T x + b_j)}
        $$

        The model learns weights $w_k$ and biases $b_k$ that separate classes in feature space. It is stronger than Naive Bayes because it learns discriminative boundaries directly.

        ### 5.4 Principal component analysis (PCA)

        PCA compresses the data into a smaller set of orthogonal directions:

        $$
        z = W^T x
        $$

        where the columns of $W$ are principal directions of maximum variance. In this notebook, PCA is used before logistic regression to test whether a more compact representation improves generalization and runtime.

        ### 5.5 Convolutional neural network (CNN)

        A convolutional layer applies learned filters over local image neighborhoods:

        $$
        h^{(l)}_{i,j,c} = \\sigma\\left((W_c * x^{(l-1)})_{i,j} + b_c\\right)
        $$

        where $*$ denotes convolution and $\\sigma$ is a nonlinearity such as ReLU.

        CNNs are appropriate here because nearby pixels matter together. A sleeve edge or a shoe outline is not well represented by treating each pixel independently.

        ### Training algorithm in pseudocode

        ```text
        for each epoch:
            for each minibatch:
                compute model outputs
                compute loss
                backpropagate gradients
                update parameters with Adam
            evaluate on validation set
            keep the checkpoint with the best validation macro F1
        ```

        The code cell below initializes all models but does not train them yet.
        """
    )
)

cells.append(
    code(
        '''
        def evaluate_predictions(y_true, y_pred):
            """Return the two summary metrics used throughout the notebook."""
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
            }


        # Classical baseline models. The pipelines ensure that scaling and PCA
        # are learned only from training data, which prevents data leakage.
        classical_models = {
            "Majority class": DummyClassifier(strategy="most_frequent"),
            "Gaussian Naive Bayes": GaussianNB(),
            "Logistic regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=300,
                            solver="lbfgs",
                            C=2.0,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            "PCA + Logistic regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=100, svd_solver="randomized", random_state=SEED)),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=300,
                            solver="lbfgs",
                            C=2.0,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
        }


        class FashionCNN(nn.Module):
            """Small LeNet-style CNN used for the deep-learning comparison."""

            def __init__(self, hidden_dim=256, dropout=0.3):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, len(CLASS_NAMES)),
                )

            def forward(self, inputs):
                features = self.features(inputs)
                return self.classifier(features)


        cnn_search_space = [
            {
                "name": "CNN-small",
                "hidden_dim": 256,
                "dropout": 0.30,
                "lr": 1e-3,
                "weight_decay": 1e-4,
            },
            {
                "name": "CNN-large",
                "hidden_dim": 384,
                "dropout": 0.40,
                "lr": 8e-4,
                "weight_decay": 1e-4,
            },
        ]
        '''
    )
)

cells.append(
    md(
        """
        ## 6. Model Training, Cross Validation, and Hyperparameter Tuning

        This section fits the models and compares them on the validation set.

        ### Training strategy

        - The classical models are fit once on the training split and evaluated on validation and test data.
        - The CNN is trained with mini-batch gradient descent using **Adam**.
        - The CNN search compares two small architectures that differ in hidden-layer width and dropout.
        - The best CNN checkpoint is selected by **validation macro F1** rather than training loss.

        ### Why macro F1?

        Accuracy can hide poor performance on individual classes. Macro F1 gives equal weight to each class, which matters in a multiclass problem where some categories are more difficult than others.
        """
    )
)

cells.append(
    code(
        """
        # Fit the classical models and store their predictions so they can be
        # compared with the CNN later.
        classical_results = []
        classical_predictions = {}

        for model_name, model in classical_models.items():
            start = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X_train_flat, y_train)
            elapsed = time.time() - start

            val_pred = model.predict(X_val_flat)
            test_pred = model.predict(X_test_flat)

            val_scores = evaluate_predictions(y_val, val_pred)
            test_scores = evaluate_predictions(y_test, test_pred)

            classical_results.append(
                {
                    "model": model_name,
                    "validation_accuracy": val_scores["accuracy"],
                    "validation_macro_f1": val_scores["macro_f1"],
                    "test_accuracy": test_scores["accuracy"],
                    "test_macro_f1": test_scores["macro_f1"],
                    "fit_seconds": elapsed,
                }
            )

            classical_predictions[model_name] = {
                "validation": val_pred,
                "test": test_pred,
            }

        classical_results_df = pd.DataFrame(classical_results).sort_values(
            by=["validation_macro_f1", "test_macro_f1"],
            ascending=False,
        )
        display(
            classical_results_df.style.format(
                {
                    "validation_accuracy": "{:.3f}",
                    "validation_macro_f1": "{:.3f}",
                    "test_accuracy": "{:.3f}",
                    "test_macro_f1": "{:.3f}",
                    "fit_seconds": "{:.1f}",
                }
            )
        )

        best_classical_name = classical_results_df.iloc[0]["model"]
        print(f"Best classical model by validation macro F1: {best_classical_name}")
        """
    )
)

cells.append(
    code(
        """
        # PCA diagnostics are shown separately so the notebook can explain what
        # dimensionality reduction is doing before moving into the CNN section.
        scaler_for_diagnostics = StandardScaler()
        X_train_scaled = scaler_for_diagnostics.fit_transform(X_train_flat)

        pca_diagnostics = PCA(n_components=150, svd_solver="randomized", random_state=SEED)
        pca_diagnostics.fit(X_train_scaled)

        cumulative_variance = np.cumsum(pca_diagnostics.explained_variance_ratio_)
        diagnostic_components = len(cumulative_variance)
        model_components = 100
        retained_variance = cumulative_variance[model_components - 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.arange(1, diagnostic_components + 1), cumulative_variance, linewidth=2)
        ax.axvline(model_components, color="black", linestyle=":", label="100 PCA components")
        ax.set_xlabel("Number of principal components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_title("PCA explained variance on the training split")
        ax.legend()
        plt.tight_layout()
        plt.show()

        print(f"Using 100 PCA components retains approximately {retained_variance:.3f} of the training variance.")
        """
    )
)

cells.append(
    code(
        '''
        # Convert the NumPy arrays into tensors and DataLoaders for PyTorch.
        X_train_tensor = torch.tensor(X_train[:, None, :, :], dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val[:, None, :, :], dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test[:, None, :, :], dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )


        def evaluate_torch_model(model, data_loader):
            """Evaluate a CNN on a loader and return loss plus classification metrics."""
            model.eval()
            all_preds = []
            all_targets = []
            total_loss = 0.0
            criterion = nn.CrossEntropyLoss()

            with torch.no_grad():
                for batch_inputs, batch_targets in data_loader:
                    batch_inputs = batch_inputs.to(DEVICE)
                    batch_targets = batch_targets.to(DEVICE)
                    logits = model(batch_inputs)
                    loss = criterion(logits, batch_targets)
                    total_loss += loss.item() * batch_inputs.size(0)

                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.append(preds)
                    all_targets.append(batch_targets.cpu().numpy())

            y_true = np.concatenate(all_targets)
            y_pred = np.concatenate(all_preds)
            return {
                "loss": total_loss / len(data_loader.dataset),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
                "y_true": y_true,
                "y_pred": y_pred,
            }


        def train_cnn(config, epochs=EPOCHS, patience=PATIENCE):
            """Train one CNN configuration and keep the best validation checkpoint."""
            model = FashionCNN(hidden_dim=config["hidden_dim"], dropout=config["dropout"]).to(DEVICE)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config["lr"],
                weight_decay=config["weight_decay"],
            )
            criterion = nn.CrossEntropyLoss()

            history = []
            best_state = copy.deepcopy(model.state_dict())
            best_val_f1 = -np.inf
            best_epoch = 0
            wait = 0
            start_time = time.time()

            for epoch in range(1, epochs + 1):
                model.train()
                running_loss = 0.0

                for batch_inputs, batch_targets in train_loader:
                    batch_inputs = batch_inputs.to(DEVICE)
                    batch_targets = batch_targets.to(DEVICE)

                    optimizer.zero_grad()
                    logits = model(batch_inputs)
                    loss = criterion(logits, batch_targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * batch_inputs.size(0)

                train_loss = running_loss / len(train_loader.dataset)
                val_metrics = evaluate_torch_model(model, val_loader)

                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"],
                    }
                )

                if val_metrics["macro_f1"] > best_val_f1 + 1e-4:
                    best_val_f1 = val_metrics["macro_f1"]
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            model.load_state_dict(best_state)
            elapsed = time.time() - start_time
            history_df = pd.DataFrame(history)
            return model, history_df, elapsed, best_epoch


        cnn_search_results = []
        cnn_histories = {}
        trained_cnn_models = {}

        for config in cnn_search_space:
            model, history_df, elapsed, best_epoch = train_cnn(config)
            val_metrics = evaluate_torch_model(model, val_loader)

            cnn_search_results.append(
                {
                    "model": config["name"],
                    "hidden_dim": config["hidden_dim"],
                    "dropout": config["dropout"],
                    "lr": config["lr"],
                    "weight_decay": config["weight_decay"],
                    "best_epoch": best_epoch,
                    "validation_accuracy": val_metrics["accuracy"],
                    "validation_macro_f1": val_metrics["macro_f1"],
                    "fit_seconds": elapsed,
                }
            )
            cnn_histories[config["name"]] = history_df
            trained_cnn_models[config["name"]] = model

        cnn_search_df = pd.DataFrame(cnn_search_results).sort_values(
            by="validation_macro_f1",
            ascending=False,
        )
        display(
            cnn_search_df.style.format(
                {
                    "dropout": "{:.2f}",
                    "lr": "{:.4f}",
                    "weight_decay": "{:.4f}",
                    "validation_accuracy": "{:.3f}",
                    "validation_macro_f1": "{:.3f}",
                    "fit_seconds": "{:.1f}",
                }
            )
        )

        best_cnn_name = cnn_search_df.iloc[0]["model"]
        best_cnn_model = trained_cnn_models[best_cnn_name]
        best_cnn_history = cnn_histories[best_cnn_name]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(best_cnn_history["epoch"], best_cnn_history["train_loss"], marker="o", label="Train loss")
        ax.plot(best_cnn_history["epoch"], best_cnn_history["val_loss"], marker="o", label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Learning curve for {best_cnn_name}")
        ax.legend()
        plt.tight_layout()
        plt.show()
        '''
    )
)

cells.append(
    code(
        '''
        display(
            Markdown(
                f"""
                ### Training and Tuning Takeaways

                - Among the classical models, **{best_classical_name}** produced the strongest validation macro F1.
                - Among the CNN configurations, **{best_cnn_name}** performed best on the validation set.
                - The learning-curve plot shows whether the CNN is still improving smoothly or whether validation performance starts to flatten out.

                This section is where model selection happens. The test set is still reserved for the next section so that final performance remains genuinely out-of-sample.
                """
            )
        )
        '''
    )
)

cells.append(
    md(
        """
        ## 7. Out-of-Sample Predictions and Performance Metrics

        Once the best models are chosen, they are applied to the **held-out test set**. This is the most important evaluation step in the notebook because it measures how well the models generalize to images they were never trained on.

        ### Metrics used

        - **Accuracy:** the fraction of all predictions that are correct.
        - **Macro F1:** the average F1 score across classes, giving each class equal weight.

        Macro F1 is especially useful here because some classes are harder than others, and a model that does well only on easy classes would be misleading.
        """
    )
)

cells.append(
    code(
        """
        cnn_val_metrics = evaluate_torch_model(best_cnn_model, val_loader)
        cnn_test_metrics = evaluate_torch_model(best_cnn_model, test_loader)

        cnn_result = pd.DataFrame(
            [
                {
                    "model": best_cnn_name,
                    "validation_accuracy": cnn_val_metrics["accuracy"],
                    "validation_macro_f1": cnn_val_metrics["macro_f1"],
                    "test_accuracy": cnn_test_metrics["accuracy"],
                    "test_macro_f1": cnn_test_metrics["macro_f1"],
                    "fit_seconds": cnn_search_df.iloc[0]["fit_seconds"],
                }
            ]
        )

        comparison_df = (
            pd.concat([classical_results_df, cnn_result], ignore_index=True)
            .sort_values(by="test_macro_f1", ascending=False)
            .reset_index(drop=True)
        )

        display(
            comparison_df.style.format(
                {
                    "validation_accuracy": "{:.3f}",
                    "validation_macro_f1": "{:.3f}",
                    "test_accuracy": "{:.3f}",
                    "test_macro_f1": "{:.3f}",
                    "fit_seconds": "{:.1f}",
                }
            )
        )
        """
    )
)

cells.append(
    md(
        """
        ## 8. Visualizing the Results and Discussing Conclusions

        A single metric table is useful, but it does not explain **where** the model succeeds or fails. The visualizations below answer more detailed questions:

        - Which class pairs are most often confused?
        - Does the CNN fix mistakes that the classical models make?
        - Are the remaining errors sensible, given the visual similarity of the classes?

        Those questions matter because good machine-learning analysis is not only about reporting a score; it is also about understanding the pattern of errors.
        """
    )
)

cells.append(
    code(
        """
        best_classical_pred = classical_predictions[best_classical_name]["test"]
        cm_classical = confusion_matrix(y_test, best_classical_pred)
        cm_cnn = confusion_matrix(cnn_test_metrics["y_true"], cnn_test_metrics["y_pred"])

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        sns.heatmap(
            cm_classical,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axes[0],
        )
        axes[0].set_title(f"Best classical model: {best_classical_name}")
        axes[0].set_xlabel("Predicted label")
        axes[0].set_ylabel("True label")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].tick_params(axis="y", rotation=0)

        sns.heatmap(
            cm_cnn,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axes[1],
        )
        axes[1].set_title(f"CNN model: {best_cnn_name}")
        axes[1].set_xlabel("Predicted label")
        axes[1].set_ylabel("True label")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].tick_params(axis="y", rotation=0)

        plt.tight_layout()
        plt.show()

        misclassified = np.where(cnn_test_metrics["y_true"] != cnn_test_metrics["y_pred"])[0][:12]
        fig, axes = plt.subplots(3, 4, figsize=(12, 10))

        for ax, idx in zip(axes.flat, misclassified):
            ax.imshow(X_test[idx], cmap="gray")
            ax.set_title(
                f"Pred: {CLASS_NAMES[cnn_test_metrics['y_pred'][idx]]}\\nTrue: {CLASS_NAMES[cnn_test_metrics['y_true'][idx]]}",
                fontsize=10,
            )
            ax.axis("off")

        for ax in axes.flat[len(misclassified):]:
            ax.axis("off")

        plt.suptitle("Sample CNN misclassifications", y=1.02)
        plt.tight_layout()
        plt.show()
        """
    )
)

cells.append(
    code(
        '''
        majority_test_f1 = comparison_df.loc[
            comparison_df["model"] == "Majority class", "test_macro_f1"
        ].iloc[0]
        cnn_test_f1 = comparison_df.loc[
            comparison_df["model"] == best_cnn_name, "test_macro_f1"
        ].iloc[0]
        classical_test_f1 = comparison_df.loc[
            comparison_df["model"] == best_classical_name, "test_macro_f1"
        ].iloc[0]

        display(
            Markdown(
                f"""
                ### Results Discussion

                - The best classical model is **{best_classical_name}** with test macro F1 **{classical_test_f1:.3f}**.
                - The best CNN is **{best_cnn_name}** with test macro F1 **{cnn_test_f1:.3f}**.
                - Relative to the strongest classical baseline, the CNN improves macro F1 by **{cnn_test_f1 - classical_test_f1:.3f}**.
                - Relative to the trivial majority-class baseline, the CNN improves macro F1 by **{cnn_test_f1 - majority_test_f1:.3f}**.

                The error plots usually show that the remaining mistakes happen among visually similar upper-body garments such as **shirts**, **pullovers**, and **coats**. That makes intuitive sense and supports the claim that the CNN is learning meaningful visual structure rather than memorizing arbitrary patterns.
                """
            )
        )
        '''
    )
)

cells.append(
    md(
        """
        ## 9. Executive Summary

        This section compresses the full notebook into a short decision-level summary.

        ### What was done

        - A sample of Fashion-MNIST images was loaded and converted into Python-readable arrays.
        - The data was normalized and prepared in two forms: flattened vectors for classical models and image tensors for the CNN.
        - Several baseline models were trained so that the CNN could be evaluated in context rather than in isolation.
        - A small CNN was then tuned on a validation set and tested on held-out data.

        ### Main finding

        The CNN outperformed the classical baselines on the held-out test set, which supports the hypothesis that preserving local spatial structure matters for image classification.

        ### Main limitation

        The modeling pipeline is strong, but the dataset is still a public benchmark rather than a sample from a personal archive. That is the main assignment-fit weakness remaining in this version.
        """
    )
)

cells.append(
    code(
        '''
        # Draw a compact pipeline diagram for the executive summary.
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.axis("off")

        steps = [
            "Dataset\\nload",
            "Sampling\\nand split",
            "Preprocess\\n(scale / flatten)",
            "Model\\ntraining",
            "Validation\\nselection",
            "Test\\nevaluation",
            "Visual\\nanalysis",
        ]
        x_positions = np.linspace(0.03, 0.87, len(steps))

        for i, (x, label) in enumerate(zip(x_positions, steps)):
            box = FancyBboxPatch(
                (x, 0.35),
                0.1,
                0.3,
                boxstyle="round,pad=0.02",
                linewidth=2,
                edgecolor="#264653",
                facecolor="#E9F1F7",
            )
            ax.add_patch(box)
            ax.text(x + 0.05, 0.5, label, ha="center", va="center", fontsize=12)
            if i < len(steps) - 1:
                ax.annotate(
                    "",
                    xy=(x_positions[i + 1], 0.5),
                    xytext=(x + 0.1, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#264653"),
                )

        plt.title("High-level modeling pipeline", pad=20)
        plt.tight_layout()
        plt.show()

        top_results = comparison_df[["model", "test_accuracy", "test_macro_f1"]].copy()
        display(top_results.style.format({"test_accuracy": "{:.3f}", "test_macro_f1": "{:.3f}"}))

        best_model_name = comparison_df.iloc[0]["model"]
        best_model_f1 = comparison_df.iloc[0]["test_macro_f1"]
        second_model_name = comparison_df.iloc[1]["model"]
        second_model_f1 = comparison_df.iloc[1]["test_macro_f1"]

        display(
            Markdown(
                f"""
                ### Executive takeaway

                The final ranking shows that **{best_model_name}** performed best on held-out data with test macro F1 **{best_model_f1:.3f}**. The runner-up was **{second_model_name}** at **{second_model_f1:.3f}**.

                The practical lesson is straightforward: once the problem is truly visual, models that preserve image structure have a clear advantage over models that see only flattened pixels.

                ### How this could be improved

                - Replace Fashion-MNIST with a labeled image sample from a real personal archive.
                - Increase the CNN search space more systematically.
                - Add data augmentation if the archive dataset is small.
                - Consider transfer learning if the final archive images are more complex than Fashion-MNIST.
                """
            )
        )
        '''
    )
)

cells.append(
    md(
        """
        ## 10. References

        - Xiao, Han, Kashif Rasul, and Roland Vollgraf. 2017. *Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747.
        - LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner. 1998. *Gradient-Based Learning Applied to Document Recognition*. *Proceedings of the IEEE* 86(11): 2278-2324.
        - [PyTorch Tutorials](https://pytorch.org/tutorials/). Used for implementation guidance on dataset loading and neural-network training loops.
        - [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). Used for the dimensionality-reduction discussion and implementation.
        - [scikit-learn LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). Used for the linear baseline setup.
        - [scikit-learn DummyClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html). Used for the majority-class baseline.
        - [scikit-learn GaussianNB documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html). Used for the Naive Bayes baseline.
        """
    )
)


nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}

with open("assignment2.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
