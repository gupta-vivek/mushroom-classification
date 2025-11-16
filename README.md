## Mushroom Classification

Machine learning project that trains a classifier to predict whether a mushroom is edible or poisonous.

### Project Structure

```
.
├── Dockerfile
├── README.md
├── data/
│   └── secondary_data.csv            # training data
│   └── secondary_data_meta.txt       # meta data   
├── models/
│   └── model.pkl                     # saved model
├── notebooks/
│   └── EDA_model.ipynb               # exploration and experimentation
├── predict.py                        # FastAPI app exposing POST /predict
├── pyproject.toml                    # project metadata & dependencies
├── test.py                           # simple client sending a sample request
├── train.py                          # training script producing models/model.pkl
└── uv.lock                           # lockfile for reproducible installs with uv
```

### Installation & Setup

```bash
uv sync
```

### Training a Model

The training script reads `data/secondary_data.csv`,performs preprocessing, trains a decision tree model within a pipeline, evaluates it, and saves the artifact to `models/model.pkl`.

Command:
```bash
uv run train.py
```

### Running the API

1) Run `predict.py`:
```bash
uv run predict.py
```
This starts Uvicorn at `http://0.0.0.0:9696`.

Endpoint:
- `POST /predict`

Request schema:
```json
{
  "cap_diameter": 15.26,
  "stem_height": 16.95,
  "stem_width": 17.09,
  "gill_color": "w",
  "does_bruise_or_bleed": "f",
  "stem_surface": "y",
  "cap_shape": "x",
  "habitat": "d",
  "gill_attachment": "e",
  "season": "w",
  "ring_type": "g",
  "cap_surface": "g",
  "cap_color": "o",
  "has_ring": "t",
  "gill_spacing": "unknown",
  "stem_color": "w"
}
```

Response example:
```json
{
  "pred": "p",
  "predict": "poisonous"
}
```

### Testing the API

Use the included `test.py` to send a sample request:
```bash
uv run test.py
```

### Docker

A `Dockerfile` is provided and uses `uv` to install dependencies and runs the API with Uvicorn.

Build the image:
```bash
docker build -t mushroom-classification .
```

Run the container (maps port 9696):
```bash
docker run --rm -p 9696:9696 mushroom-classification
```
