# General Loader Principle

Loaders are instances that handles data batching from input datasets to construct a dictionary based on `torchio`. All dataloaders should have a full definition for training and inference dataloading machanisms.

```mermaid
flowchart TD
  TD[("Target Data <br> (Optional for Inference)")]
  ID[(Input Data)]
  ID & TD --> L(PMIDataLoader)
  Main(Controller) --> |Request Data|L
  L --> |Return torchio.Queue|Main
```

In essence, the loader is the API to request a `torchio.Queue` that will be used by the `Solver` or `Inferencer`.  Different situation calls for different prepration workflow.

# Examples

## PMIImageDataLoader

This loader is the typical loader you will use for img2img task. The loading mechanism requires an `ImageDataSet` input and another `ImageDataSet` target.

The key reason is because images are often of different raw sizes, and theres a need to extract patches from these images. Therefore, there's a `sampler` parameter in this loader that is not anticipated in other kind of non-imaging loaders.

```mermaid
graph LR
A[PMIImageDataLoader] --> B["Initialize with config (cfg)"]
B --> C[Check configuration <br> using _read_config]
C -->|Sampler: weighted| D[Check probmap_dir <br> and sampler_kwargs]
C -->|Sampler: uniform| E[Check patch_size in sampler_kwargs]
C -->|Sampler: grid| F[Check patch_size and <br>  patch_overlap in sampler_kwargs]
C -->|No sampler| G[Assume whole image is sampled]

B --> H[Load dataset <br> for training or inference]
H --> I[_load_data_set_training]
H --> J[_load_data_set_inference]

I --> K[Prepare data <br> using _prepare_data]
K --> L[Read input <br> images using _read_image]
K --> M[Read ground-truth <br> data using _load_gt_data]
K --> N[Read masks<br>  using _read_image]
K --> O[Prepare probability map <br> using _prepare_probmap]

I --> P[Pack data into subjects <br> using _pack_data_into_subjects]
P --> Q[Apply transform <br> created using _create_transform]

I --> R[Create queue <br> using _create_queue]
R -->|Sampler specified| S[Create tio.Queue <br> or CallbackQueue]
R -->|No sampler| T[Use UniformSampler <br> with patch <br> size = image shape]

J --> U[Override samples_per_volume<br>  if inf_samples_per_vol is specified]
J --> V[Load data using <br> _load_data_set_training <br> with exclude_augment flag]

V -->|Force augment| W[Enable data <br> augmentation]
V -->|No force augment| X[Disable data <br> augmentation]

R --> Y[Return queue or queue <br> + sampler for inference]
```



