<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>

<h1 align="center">
  PyKEEN
</h1>

<p align="center">
  <a href="https://github.com/pykeen/pykeen/actions">
    <img src="https://github.com/pykeen/pykeen/workflows/Tests%20master/badge.svg"
         alt="GitHub Actions">
  </a>

  <a href='https://opensource.org/licenses/MIT'>
    <img src='https://img.shields.io/badge/License-MIT-blue.svg' alt='License'/>
  </a>

  <a href="https://zenodo.org/badge/latestdoi/242672435">
    <img src="https://zenodo.org/badge/242672435.svg" alt="DOI">
  </a>

  <a href="https://optuna.org">
    <img src="https://img.shields.io/badge/Optuna-integrated-blue" alt="Optuna integrated" height="20">
  </a>
</p>

<p align="center">
    <b>PyKEEN</b> (<b>P</b>ython <b>K</b>nowl<b>E</b>dge <b>E</b>mbeddi<b>N</b>gs) is a Python package designed to
    train and evaluate knowledge graph embedding models (incorporating multi-modal information).
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#datasets-25">Datasets</a> •
  <a href="#models-25">Models</a> •
  <a href="#supporters">Support</a> •
  <a href="#citation">Citation</a>
</p>

## Installation ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pykeen) ![PyPI](https://img.shields.io/pypi/v/pykeen)

The latest stable version of PyKEEN can be downloaded and installed from
[PyPI](https://pypi.org/project/pykeen) with:

```bash
$ pip install pykeen
```

The latest version of PyKEEN can be installed directly from the
source on [GitHub](https://github.com/pykeen/pykeen) with:

```bash
pip install git+https://github.com/pykeen/pykeen.git
```

More information about installation (e.g., development mode, Windows installation, extras)
can be found in the [installation documentation](https://pykeen.readthedocs.io/en/latest/installation.html).

## Quickstart [![Documentation Status](https://readthedocs.org/projects/pykeen/badge/?version=latest)](https://pykeen.readthedocs.io/en/latest/?badge=latest)

This example shows how to train a model on a dataset and test on another dataset.

The fastest way to get up and running is to use the pipeline function. It
provides a high-level entry into the extensible functionality of this package.
The following example shows how to train and evaluate the [TransE](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.TransE.html#pykeen.models.TransE)
model on the [Nations](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.Nations.html#pykeen.datasets.Nations)
dataset. By default, the training loop uses the [stochastic local closed world assumption (sLCWA)](https://pykeen.readthedocs.io/en/latest/reference/training.html#pykeen.training.SLCWATrainingLoop)
training approach and evaluates with [rank-based evaluation](https://pykeen.readthedocs.io/en/latest/reference/evaluation/rank_based.html#pykeen.evaluation.RankBasedEvaluator).

```python
from pykeen.pipeline import pipeline

result = pipeline(
    model='TransE',
    dataset='nations',
)
```

The results are returned in an instance of the [PipelineResult](https://pykeen.readthedocs.io/en/latest/reference/pipeline.html#pykeen.pipeline.PipelineResult)
dataclass that has attributes for the trained model, the training loop, the evaluation, and more. See the tutorials on
[understanding the evaluation](https://pykeen.readthedocs.io/en/latest/tutorial/understanding_evaluation.html)
and [making novel link predictions](https://pykeen.readthedocs.io/en/latest/tutorial/making_predictions.html).

PyKEEN is extensible such that:

- Each model has the same API, so anything from ``pykeen.models`` can be dropped in
- Each training loop has the same API, so ``pykeen.training.LCWATrainingLoop`` can be dropped in
- Triples factories can be generated by the user with ``from pykeen.triples.TriplesFactory``

The full documentation can be found at https://pykeen.readthedocs.io.

## Implementation

Below are the models, datasets, training modes, evaluators, and metrics implemented
in ``pykeen``.

### Datasets (25)

The citation for each dataset corresponds to either the paper describing the dataset,
the first paper published using the dataset with knowledge graph embedding models,
or the URL for the dataset if neither of the first two are available.

| Name                               | Documentation                                                                                                     | Citation                                                                                                                |   Entities |   Relations |   Triples |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|------------|-------------|-----------|
| Clinical Knowledge Graph           | [`pykeen.datasets.CKG`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.CKG.html)                     | [Santos *et al*., 2020](https://doi.org/10.1101/2020.05.09.084897)                                                      |    7617419 |          11 |  26691525 |
| CoDEx (large)                      | [`pykeen.datasets.CoDExLarge`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.CoDExLarge.html)       | [Safavi *et al*., 2020](https://arxiv.org/abs/2009.07810)                                                               |      77951 |          69 |    612437 |
| CoDEx (medium)                     | [`pykeen.datasets.CoDExMedium`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.CoDExMedium.html)     | [Safavi *et al*., 2020](https://arxiv.org/abs/2009.07810)                                                               |      17050 |          51 |    206205 |
| CoDEx (small)                      | [`pykeen.datasets.CoDExSmall`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.CoDExSmall.html)       | [Safavi *et al*., 2020](https://arxiv.org/abs/2009.07810)                                                               |       2034 |          42 |     36543 |
| ConceptNet                         | [`pykeen.datasets.ConceptNet`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.ConceptNet.html)       | [Speer *et al*., 2017](https://arxiv.org/abs/1612.03975)                                                                |   28370083 |          50 |  34074917 |
| Countries                          | [`pykeen.datasets.Countries`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.Countries.html)         | [Bouchard *et al*., 2015](https://www.aaai.org/ocs/index.php/SSS/SSS15/paper/view/10257/10026)                          |        271 |           2 |      1158 |
| Commonsense Knowledge Graph        | [`pykeen.datasets.CSKG`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.CSKG.html)                   | [Ilievski *et al*., 2020](http://arxiv.org/abs/2012.11490)                                                              |    2087833 |          58 |   4598728 |
| DB100K                             | [`pykeen.datasets.DB100K`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.DB100K.html)               | [Ding *et al*., 2018](https://arxiv.org/abs/1805.02408)                                                                 |      99604 |         470 |    697479 |
| DBpedia50                          | [`pykeen.datasets.DBpedia50`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.DBpedia50.html)         | [Shi *et al*., 2017](https://arxiv.org/abs/1711.03438)                                                                  |      24624 |         351 |     34421 |
| Drug Repositioning Knowledge Graph | [`pykeen.datasets.DRKG`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.DRKG.html)                   | [`gnn4dr/DRKG`](https://github.com/gnn4dr/DRKG)                                                                         |      97238 |         107 |   5874257 |
| FB15k                              | [`pykeen.datasets.FB15k`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.FB15k.html)                 | [Bordes *et al*., 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) |      14951 |        1345 |    592213 |
| FB15k-237                          | [`pykeen.datasets.FB15k237`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.FB15k237.html)           | [Toutanova *et al*., 2015](https://www.aclweb.org/anthology/W15-4007/)                                                  |      14505 |         237 |    310079 |
| Hetionet                           | [`pykeen.datasets.Hetionet`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.Hetionet.html)           | [Himmelstein *et al*., 2017](https://doi.org/10.7554/eLife.26726)                                                       |      45158 |          24 |   2250197 |
| Kinships                           | [`pykeen.datasets.Kinships`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.Kinships.html)           | [Kemp *et al*., 2006](https://www.aaai.org/Papers/AAAI/2006/AAAI06-061.pdf)                                             |        104 |          25 |     10686 |
| Nations                            | [`pykeen.datasets.Nations`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.Nations.html)             | [`ZhenfengLei/KGDatasets`](https://github.com/ZhenfengLei/KGDatasets)                                                   |         14 |          55 |      1992 |
| OGB BioKG                          | [`pykeen.datasets.OGBBioKG`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.OGBBioKG.html)           | [Hu *et al*., 2020](https://arxiv.org/abs/2005.00687)                                                                   |      45085 |          51 |   5088433 |
| OGB WikiKG                         | [`pykeen.datasets.OGBWikiKG`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.OGBWikiKG.html)         | [Hu *et al*., 2020](https://arxiv.org/abs/2005.00687)                                                                   |    2500604 |         535 |  17137181 |
| OpenBioLink                        | [`pykeen.datasets.OpenBioLink`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.OpenBioLink.html)     | [Breit *et al*., 2020](https://doi.org/10.1093/bioinformatics/btaa274)                                                  |     180992 |          28 |   4563407 |
| OpenBioLink (F1)                   | [`pykeen.datasets.OpenBioLinkF1`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.OpenBioLinkF1.html) | [`PyKEEN/pykeen-openbiolink-benchmark`](https://github.com/PyKEEN/pykeen-openbiolink-benchmark)                         |     116425 |          19 |   1716703 |
| OpenBioLink (F2)                   | [`pykeen.datasets.OpenBioLinkF2`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.OpenBioLinkF2.html) | [`PyKEEN/pykeen-openbiolink-benchmark`](https://github.com/PyKEEN/pykeen-openbiolink-benchmark)                         |     110628 |          17 |    734925 |
| OpenBioLink                        | [`pykeen.datasets.OpenBioLinkLQ`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.OpenBioLinkLQ.html) | [Breit *et al*., 2020](https://doi.org/10.1093/bioinformatics/btaa274)                                                  |     480876 |          32 |  27320889 |
| Unified Medical Language System    | [`pykeen.datasets.UMLS`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.UMLS.html)                   | [`ZhenfengLei/KGDatasets`](https://github.com/ZhenfengLei/KGDatasets)                                                   |        135 |          46 |      6529 |
| WordNet-18                         | [`pykeen.datasets.WN18`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.WN18.html)                   | [Bordes *et al*., 2014](https://arxiv.org/abs/1301.3485)                                                                |      40943 |          18 |    151442 |
| WordNet-18 (RR)                    | [`pykeen.datasets.WN18RR`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.WN18RR.html)               | [Toutanova *et al*., 2015](https://www.aclweb.org/anthology/W15-4007/)                                                  |      40559 |          11 |     92583 |
| YAGO3-10                           | [`pykeen.datasets.YAGO310`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.YAGO310.html)             | [Mahdisoltani *et al*., 2015](http://service.tsi.telecom-paristech.fr/cgi-bin//valipub_download.cgi?dId=284)            |     123143 |          37 |   1089000 |

### Models (25)

| Name                | Reference                                                                                                                 | Citation                                                                                                                |
|---------------------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| ComplEx             | [`pykeen.models.ComplEx`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.ComplEx.html)                         | [Trouillon *et al.*, 2016](https://arxiv.org/abs/1606.06357)                                                            |
| ComplExLiteral      | [`pykeen.models.ComplExLiteral`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.ComplExLiteral.html)           | [Kristiadi *et al.*, 2018](https://arxiv.org/abs/1802.00934)                                                            |
| ConvE               | [`pykeen.models.ConvE`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.ConvE.html)                             | [Dettmers *et al.*, 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366)                              |
| ConvKB              | [`pykeen.models.ConvKB`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.ConvKB.html)                           | [Nguyen *et al.*, 2018](https://www.aclweb.org/anthology/N18-2053)                                                      |
| DistMult            | [`pykeen.models.DistMult`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.DistMult.html)                       | [Yang *et al.*, 2014](https://arxiv.org/abs/1412.6575)                                                                  |
| DistMultLiteral     | [`pykeen.models.DistMultLiteral`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.DistMultLiteral.html)         | [Kristiadi *et al.*, 2018](https://arxiv.org/abs/1802.00934)                                                            |
| ERMLP               | [`pykeen.models.ERMLP`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.ERMLP.html)                             | [Dong *et al.*, 2014](https://dl.acm.org/citation.cfm?id=2623623)                                                       |
| ERMLPE              | [`pykeen.models.ERMLPE`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.ERMLPE.html)                           | [Sharifzadeh *et al.*, 2019](https://github.com/pykeen/pykeen)                                                          |
| HolE                | [`pykeen.models.HolE`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.HolE.html)                               | [Nickel *et al.*, 2016](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828)                      |
| KG2E                | [`pykeen.models.KG2E`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.KG2E.html)                               | [He *et al.*, 2015](https://dl.acm.org/doi/10.1145/2806416.2806502)                                                     |
| MuRE                | [`pykeen.models.MuRE`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.MuRE.html)                               | [Balažević *et al.*, 2019](https://arxiv.org/abs/1905.09791)                                                            |
| NTN                 | [`pykeen.models.NTN`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.NTN.html)                                 | [Socher *et al.*, 2013](https://dl.acm.org/doi/10.5555/2999611.2999715)                                                 |
| PairRE              | [`pykeen.models.PairRE`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.PairRE.html)                           | [Chao *et al.*, 2020](http://arxiv.org/abs/2011.03798)                                                                  |
| ProjE               | [`pykeen.models.ProjE`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.ProjE.html)                             | [Shi *et al.*, 2017](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14279)                                   |
| RESCAL              | [`pykeen.models.RESCAL`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.RESCAL.html)                           | [Nickel *et al.*, 2011](http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf)                                      |
| RGCN                | [`pykeen.models.RGCN`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.RGCN.html)                               | [Schlichtkrull *et al.*, 2018](https://arxiv.org/pdf/1703.06103)                                                        |
| RotatE              | [`pykeen.models.RotatE`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.RotatE.html)                           | [Sun *et al.*, 2019](https://arxiv.org/abs/1902.10197v1)                                                                |
| SimplE              | [`pykeen.models.SimplE`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.SimplE.html)                           | [Kazemi *et al.*, 2018](https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs)     |
| StructuredEmbedding | [`pykeen.models.StructuredEmbedding`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.StructuredEmbedding.html) | [Bordes *et al.*, 2011](http://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/download/3659/3898)                         |
| TransD              | [`pykeen.models.TransD`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.TransD.html)                           | [Ji *et al.*, 2015](http://www.aclweb.org/anthology/P15-1067)                                                           |
| TransE              | [`pykeen.models.TransE`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.TransE.html)                           | [Bordes *et al.*, 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) |
| TransH              | [`pykeen.models.TransH`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.TransH.html)                           | [Wang *et al.*, 2014](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8531/8546)                          |
| TransR              | [`pykeen.models.TransR`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.TransR.html)                           | [Lin *et al.*, 2015](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523/)                           |
| TuckER              | [`pykeen.models.TuckER`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.TuckER.html)                           | [Balažević *et al.*, 2019](https://arxiv.org/abs/1901.09590)                                                            |
| UnstructuredModel   | [`pykeen.models.UnstructuredModel`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.UnstructuredModel.html)     | [Bordes *et al.*, 2014](https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf)                          |

### Losses (7)

| Name            | Reference                                                                                                                 | Description                                                                                       |
|-----------------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| bceaftersigmoid | [`pykeen.losses.BCEAfterSigmoidLoss`](https://pykeen.readthedocs.io/en/latest/api/pykeen.losses.BCEAfterSigmoidLoss.html) | A module for the numerically unstable version of explicit Sigmoid + BCE loss.                     |
| bcewithlogits   | [`pykeen.losses.BCEWithLogitsLoss`](https://pykeen.readthedocs.io/en/latest/api/pykeen.losses.BCEWithLogitsLoss.html)     | A module for the binary cross entropy loss.                                                       |
| crossentropy    | [`pykeen.losses.CrossEntropyLoss`](https://pykeen.readthedocs.io/en/latest/api/pykeen.losses.CrossEntropyLoss.html)       | A module for the cross entopy loss that evaluates the cross entropy after softmax output.         |
| marginranking   | [`pykeen.losses.MarginRankingLoss`](https://pykeen.readthedocs.io/en/latest/api/pykeen.losses.MarginRankingLoss.html)     | A module for the margin ranking loss.                                                             |
| mse             | [`pykeen.losses.MSELoss`](https://pykeen.readthedocs.io/en/latest/api/pykeen.losses.MSELoss.html)                         | A module for the mean square error loss.                                                          |
| nssa            | [`pykeen.losses.NSSALoss`](https://pykeen.readthedocs.io/en/latest/api/pykeen.losses.NSSALoss.html)                       | An implementation of the self-adversarial negative sampling loss function proposed by [sun2019]_. |
| softplus        | [`pykeen.losses.SoftplusLoss`](https://pykeen.readthedocs.io/en/latest/api/pykeen.losses.SoftplusLoss.html)               | A module for the softplus loss.                                                                   |

### Regularizers (5)

| Name     | Reference                                                                                                                             | Description                                              |
|----------|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| combined | [`pykeen.regularizers.CombinedRegularizer`](https://pykeen.readthedocs.io/en/latest/api/pykeen.regularizers.CombinedRegularizer.html) | A convex combination of regularizers.                    |
| lp       | [`pykeen.regularizers.LpRegularizer`](https://pykeen.readthedocs.io/en/latest/api/pykeen.regularizers.LpRegularizer.html)             | A simple L_p norm based regularizer.                     |
| no       | [`pykeen.regularizers.NoRegularizer`](https://pykeen.readthedocs.io/en/latest/api/pykeen.regularizers.NoRegularizer.html)             | A regularizer which does not perform any regularization. |
| powersum | [`pykeen.regularizers.PowerSumRegularizer`](https://pykeen.readthedocs.io/en/latest/api/pykeen.regularizers.PowerSumRegularizer.html) | A simple x^p based regularizer.                          |
| transh   | [`pykeen.regularizers.TransHRegularizer`](https://pykeen.readthedocs.io/en/latest/api/pykeen.regularizers.TransHRegularizer.html)     | A regularizer for the soft constraints in TransH.        |

### Optimizers (6)

| Name     | Reference                                                                                 | Description                                                             |
|----------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| adadelta | [`torch.optim.Adadelta`](https://pytorch.org/docs/stable/optim.html#torch.optim.Adadelta) | Implements Adadelta algorithm.                                          |
| adagrad  | [`torch.optim.Adagrad`](https://pytorch.org/docs/stable/optim.html#torch.optim.Adagrad)   | Implements Adagrad algorithm.                                           |
| adam     | [`torch.optim.Adam`](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)         | Implements Adam algorithm.                                              |
| adamax   | [`torch.optim.Adamax`](https://pytorch.org/docs/stable/optim.html#torch.optim.Adamax)     | Implements Adamax algorithm (a variant of Adam based on infinity norm). |
| adamw    | [`torch.optim.AdamW`](https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW)       | Implements AdamW algorithm.                                             |
| sgd      | [`torch.optim.SGD`](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)           | Implements stochastic gradient descent (optionally with momentum).      |

### Training Loops (2)

| Name   | Reference                                                                                                                                | Description                                                                               |
|--------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| lcwa   | [`pykeen.training.LCWATrainingLoop`](https://pykeen.readthedocs.io/en/latest/reference/training.html#pykeen.training.LCWATrainingLoop)   | A training loop that uses the local closed world assumption training approach.            |
| slcwa  | [`pykeen.training.SLCWATrainingLoop`](https://pykeen.readthedocs.io/en/latest/reference/training.html#pykeen.training.SLCWATrainingLoop) | A training loop that uses the stochastic local closed world assumption training approach. |

### Negative Samplers (2)

| Name      | Reference                                                                                                                               | Description                                                                            |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| basic     | [`pykeen.sampling.BasicNegativeSampler`](https://pykeen.readthedocs.io/en/latest/api/pykeen.sampling.BasicNegativeSampler.html)         | A basic negative sampler.                                                              |
| bernoulli | [`pykeen.sampling.BernoulliNegativeSampler`](https://pykeen.readthedocs.io/en/latest/api/pykeen.sampling.BernoulliNegativeSampler.html) | An implementation of the Bernoulli negative sampling approach proposed by [wang2014]_. |

### Stoppers (2)

| Name   | Reference                                                                                                                      | Description                   |
|--------|--------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| early  | [`pykeen.stoppers.EarlyStopper`](https://pykeen.readthedocs.io/en/latest/reference/stoppers.html#pykeen.stoppers.EarlyStopper) | A harness for early stopping. |
| nop    | [`pykeen.stoppers.NopStopper`](https://pykeen.readthedocs.io/en/latest/reference/stoppers.html#pykeen.stoppers.NopStopper)     | A stopper that does nothing.  |

### Evaluators (2)

| Name      | Reference                              | Description                                   |
|-----------|----------------------------------------|-----------------------------------------------|
| rankbased | `pykeen.evaluation.RankBasedEvaluator` | A rank-based evaluator for KGE models.        |
| sklearn   | `pykeen.evaluation.SklearnEvaluator`   | An evaluator that uses a Scikit-learn metric. |

### Metrics (6)

| Metric                  | Description                                                                                                        | Evaluator   | Reference                                  |
|-------------------------|--------------------------------------------------------------------------------------------------------------------|-------------|--------------------------------------------|
| Adjusted Mean Rank      | The mean over all chance-adjusted ranks: mean_i (2r_i / (num_entities+1)). Lower is better.                        | rankbased   | `pykeen.evaluation.RankBasedMetricResults` |
| Average Precision Score | The area under the precision-recall curve, between [0.0, 1.0]. Higher is better.                                   | sklearn     | `pykeen.evaluation.SklearnMetricResults`   |
| Hits At K               | The hits at k for different values of k, i.e. the relative frequency of ranks not larger than k. Higher is better. | rankbased   | `pykeen.evaluation.RankBasedMetricResults` |
| Mean Rank               | The mean over all ranks: mean_i r_i. Lower is better.                                                              | rankbased   | `pykeen.evaluation.RankBasedMetricResults` |
| Mean Reciprocal Rank    | The mean over all reciprocal ranks: mean_i (1/r_i). Higher is better.                                              | rankbased   | `pykeen.evaluation.RankBasedMetricResults` |
| Roc Auc Score           | The area under the ROC curve between [0.0, 1.0]. Higher is better.                                                 | sklearn     | `pykeen.evaluation.SklearnMetricResults`   |

### Trackers (5)

| Name    | Reference                                                                                                                       | Description                            |
|---------|---------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| csv     | [`pykeen.trackers.CSVResultTracker`](https://pykeen.readthedocs.io/en/latest/api/pykeen.trackers.CSVResultTracker.html)         | Tracking results to a CSV file.        |
| json    | [`pykeen.trackers.JSONResultTracker`](https://pykeen.readthedocs.io/en/latest/api/pykeen.trackers.JSONResultTracker.html)       | Tracking results to a JSON lines file. |
| mlflow  | [`pykeen.trackers.MLFlowResultTracker`](https://pykeen.readthedocs.io/en/latest/api/pykeen.trackers.MLFlowResultTracker.html)   | A tracker for MLflow.                  |
| neptune | [`pykeen.trackers.NeptuneResultTracker`](https://pykeen.readthedocs.io/en/latest/api/pykeen.trackers.NeptuneResultTracker.html) | A tracker for Neptune.ai.              |
| wandb   | [`pykeen.trackers.WANDBResultTracker`](https://pykeen.readthedocs.io/en/latest/api/pykeen.trackers.WANDBResultTracker.html)     | A tracker for Weights and Biases.      |

## Hyper-parameter Optimization

### Samplers (3)

| Name   | Reference                                                                                                                         | Description                                                     |
|--------|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| grid   | [`optuna.samplers.GridSampler`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.GridSampler.html)     | Sampler using grid search.                                      |
| random | [`optuna.samplers.RandomSampler`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.RandomSampler.html) | Sampler using random sampling.                                  |
| tpe    | [`optuna.samplers.TPESampler`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html)       | Sampler using TPE (Tree-structured Parzen Estimator) algorithm. |

Any sampler class extending the [optuna.samplers.BaseSampler](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.BaseSampler.html#optuna.samplers.BaseSampler),
such as their sampler implementing the [CMA-ES](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html#optuna.samplers.CmaEsSampler)
algorithm, can also be used.

## Experimentation

### Reproduction

PyKEEN includes a set of curated experimental settings for reproducing past landmark
experiments. They can be accessed and run like:

```bash
pykeen experiments reproduce tucker balazevic2019 fb15k
```

Where the three arguments are the model name, the reference, and the dataset.
The output directory can be optionally set with `-d`.

### Ablation

PyKEEN includes the ability to specify ablation studies using the
hyper-parameter optimization module. They can be run like:

```bash
pykeen experiments ablation ~/path/to/config.json
```

### Large-scale Reproducibility and Benchmarking Study

We used PyKEEN to perform a large-scale reproducibility and benchmarking study which are described in
[our article](https://arxiv.org/abs/2006.13365):

```bibtex
@article{ali2020benchmarking,
  title={Bringing Light Into the Dark: A Large-scale Evaluation of Knowledge Graph Embedding Models Under a Unified Framework},
  author={Ali, Mehdi and Berrendorf, Max and Hoyt, Charles Tapley and Vermue, Laurent and Galkin, Mikhail and Sharifzadeh, Sahand and Fischer, Asja and Tresp, Volker and Lehmann, Jens},
  journal={arXiv preprint arXiv:2006.13365},
  year={2020}
}
```

We have made all code, experimental configurations, results, and analyses that lead to our interpretations available
at https://github.com/pykeen/benchmarking.

## Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. 
See [CONTRIBUTING.md](/CONTRIBUTING.md) for more information on getting involved.

## Acknowledgements

### Supporters

This project has been supported by several organizations (in alphabetical order):

- [Bayer](https://www.bayer.com/)
- [CoronaWhy](https://www.coronawhy.org/)
- [Enveda Biosciences](https://www.envedabio.com/)
- [Fraunhofer Institute for Algorithms and Scientific Computing](https://www.scai.fraunhofer.de)
- [Fraunhofer Institute for Intelligent Analysis and Information Systems](https://www.iais.fraunhofer.de)
- [Fraunhofer Center for Machine Learning](https://www.cit.fraunhofer.de/de/zentren/maschinelles-lernen.html)
- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)
- [Laboratory of Systems Pharmacology, Harvard Medical School](https://hits.harvard.edu)
- [Ludwig-Maximilians-Universität München](https://www.en.uni-muenchen.de/index.html)
- [Munich Center for Machine Learning (MCML)](https://mcml.ai/)
- [Siemens](https://new.siemens.com/global/en.html)
- [Smart Data Analytics Research Group (University of Bonn & Fraunhofer IAIS)](https://sda.tech)
- [Technical University of Denmark - DTU Compute - Section for Cognitive Systems](https://www.compute.dtu.dk/english/research/research-sections/cogsys)
- [Technical University of Denmark - DTU Compute - Section for Statistics and Data Analysis](https://www.compute.dtu.dk/english/research/research-sections/stat)
- [University of Bonn](https://www.uni-bonn.de/)

### Logo

The PyKEEN logo was designed by Carina Steinborn.

## Citation

If you have found PyKEEN useful in your work, please consider citing [our article](https://arxiv.org/abs/2007.14175):

```bibtex
@article{ali2021pykeen,
    author = {Ali, Mehdi and Berrendorf, Max and Hoyt, Charles Tapley and Vermue, Laurent and Sharifzadeh, Sahand and Tresp, Volker and Lehmann, Jens},
    journal = {Journal of Machine Learning Research},
    number = {82},
    pages = {1--6},
    title = {{PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings}}},
    url = {http://jmlr.org/papers/v22/20-825.html},
    volume = {22},
    year = {2021}
}
```
