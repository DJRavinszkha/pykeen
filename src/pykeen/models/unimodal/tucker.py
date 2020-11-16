# -*- coding: utf-8 -*-

"""Implementation of TuckEr."""

from typing import Any, ClassVar, Mapping, Optional

from .. import SingleVectorEmbeddingModel
from ...losses import BCEAfterSigmoidLoss, Loss
from ...nn.emb import EmbeddingSpecification
from ...nn.init import xavier_normal_
from ...nn.modules import TuckerInteraction
from ...triples import TriplesFactory
from ...typing import DeviceHint

__all__ = [
    'TuckER',
]


class TuckER(SingleVectorEmbeddingModel):
    r"""An implementation of TuckEr from [balazevic2019]_.

    TuckER is a linear model that is based on the tensor factorization method Tucker in which a three-mode tensor
    $\mathfrak{X} \in \mathbb{R}^{I \times J \times K}$ is decomposed into a set of factor matrices
    $\textbf{A} \in \mathbb{R}^{I \times P}$, $\textbf{B} \in \mathbb{R}^{J \times Q}$, and
    $\textbf{C} \in \mathbb{R}^{K \times R}$ and a core tensor
    $\mathfrak{Z} \in \mathbb{R}^{P \times Q \times R}$ (of lower rank):

    .. math::

        \mathfrak{X} \approx \mathfrak{Z} \times_1 \textbf{A} \times_2 \textbf{B} \times_3 \textbf{C}

    where $\times_n$ is the tensor product, with $n$ denoting along which mode the tensor product is computed.
    In TuckER, a knowledge graph is considered as a binary tensor which is factorized using the Tucker factorization
    where $\textbf{E} = \textbf{A} = \textbf{C} \in \mathbb{R}^{n_{e} \times d_e}$ denotes the entity embedding
    matrix, $\textbf{R} = \textbf{B} \in \mathbb{R}^{n_{r} \times d_r}$ represents the relation embedding matrix,
    and $\mathfrak{W} = \mathfrak{Z} \in \mathbb{R}^{d_e \times d_r \times d_e}$ is the *core tensor* that
    indicates the extent of interaction between the different factors. The interaction model is defined as:

    .. math::

        f(h,r,t) = \mathfrak{W} \times_1 \textbf{h} \times_2 \textbf{r} \times_3 \textbf{t}

    where $\textbf{h},\textbf{t}$ correspond to rows of $\textbf{E}$ and $\textbf{r}$ to a row of $\textbf{R}$.

    .. seealso::

       - Official implementation: https://github.com/ibalazevic/TuckER
       - pykg2vec implementation of TuckEr https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TuckER.py
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default = dict(
        embedding_dim=dict(type=int, low=50, high=300, q=50),
        relation_dim=dict(type=int, low=30, high=200, q=25),
        dropout_0=dict(type=float, low=0.1, high=0.4),
        dropout_1=dict(type=float, low=0.1, high=0.5),
        dropout_2=dict(type=float, low=0.1, high=0.6),
    )
    #: The default loss function class
    loss_default = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 200,
        automatic_memory_optimization: Optional[bool] = None,
        relation_dim: Optional[int] = None,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        dropout_0: float = 0.3,
        dropout_1: float = 0.4,
        dropout_2: float = 0.5,
        apply_batch_normalization: bool = True,
    ) -> None:
        """Initialize the model.

        The dropout values correspond to the following dropouts in the model's score function:

            DO_2(BN(DO_0(BN(h)) x_1 DO_1(W x_2 r))) x_3 t

        where h,r,t are the head, relation, and tail embedding, W is the core tensor, x_i denotes the tensor
        product along the i-th mode, BN denotes batch normalization, and DO dropout.
        """
        super().__init__(
            triples_factory=triples_factory,
            interaction=TuckerInteraction(
                embedding_dim=embedding_dim,
                relation_dim=relation_dim,
                dropout_0=dropout_0,
                dropout_1=dropout_1,
                dropout_2=dropout_2,
                apply_batch_normalization=apply_batch_normalization,
            ),
            embedding_dim=embedding_dim,
            relation_dim=relation_dim,
            automatic_memory_optimization=automatic_memory_optimization,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            embedding_specification=EmbeddingSpecification(
                initializer=xavier_normal_,
            ),
            relation_embedding_specification=EmbeddingSpecification(
                initializer=xavier_normal_,
            ),
        )
