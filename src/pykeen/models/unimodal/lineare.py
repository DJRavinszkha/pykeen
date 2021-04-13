# -*- coding: utf-8 -*-

"""Implementation of LineaRE."""

from functools import partial
from typing import Any, ClassVar, Mapping, Optional

import torch
import torch.autograd
import torch.nn.init

from ..base import EntityRelationEmbeddingModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...losses import Loss
from ...nn.emb import Embedding, EmbeddingSpecification
from ...nn.init import zeros_
from ...regularizers import Regularizer
from ...triples import TriplesFactory
from ...typing import Constrainer, DeviceHint, Hint, Initializer
from ...utils import clamp_norm

__all__ = [
    'LineaRE',
]


def _projection_initializer(
    x: torch.FloatTensor,
    num_relations: int,
    embedding_dim: int,
    relation_dim: int,
) -> torch.FloatTensor:
    """Initialize by Glorot."""
    return torch.nn.init.xavier_uniform_(x.view(num_relations, embedding_dim, relation_dim)).view(x.shape)


class LineaRE(EntityRelationEmbeddingModel):
    r"""An implementation of LineaRE from [Peng2020]_.

    Linear Regression Embedding (LineaRE) is closely related to TransE and TransR however is able to take
    into account complex mapping properties similarly to DistMult or ComplEx. In addition to this,
    it is also able to take four relation connectivity patterns into account similarly to RotatE

    Each entity in the KG is represented a low dimensional vector h or t.
    Each relation in the KG is represented as two weight vectors w1 and w2. 

    Given a perfect triplet h, r, t:
        w1 o h = w2 o t

        where o is the Hadamard product.
        
    .. seealso::

       - LineaRE
         <https://github.com/pengyanhui/LineaRE>`_
 ---
    citation:
        author: Peng
        year: 2020
        link: https://arxiv.org/pdf/2004.10037.pdf
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        relation_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        triples_factory: TriplesFactory,
        embedding_dim: int = 50,
        relation_dim: int = 30,
        scoring_fct_norm: int = 1,
        loss: Optional[Loss] = None,
        preferred_device: DeviceHint = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__(
            triples_factory=triples_factory,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=relation_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
            ),
        )
        self.scoring_fct_norm = scoring_fct_norm

        # embeddings

        self.wrh = Embedding.init_with_device(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=relation_dim,
            device=self.device,
            initializer=partial(
                _projection_initializer,
                num_relations=self.num_relations,
                embedding_dim=self.embedding_dim,
                relation_dim=self.relation_dim,
            ),
        self.wrt = Embedding.init_with_device(
            num_embeddings=triples_factory.num_relations,
            embedding_dim=relation_dim,
            device=self.device,
            initializer=partial(
                _projection_initializer,
                num_relations=self.num_relations,
                embedding_dim=self.embedding_dim,
                relation_dim=self.relation_dim,
            ),
        zeros_(self.wrh.weight),
        zeros_(self.wrt.weight),
        )

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
        wh: torch.FloatTensor,
        wt: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function of LineaRE for given embeddings.
            Because this is based on a linear regression, not many constraints
            or further processing needs to be conducted
            """
        # evaluate score function, shape: (b, e)
        return -torch.norm(wh * h + r - wt * t, dim=-1) ** 2

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(indices=hrt_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings(indices=hrt_batch[:, 2]).unsqueeze(dim=1)
        wh = self.wrh(indices=hrt_batch[:, 1]).view(-1, self.embedding_dim, self.relation_dim)
        wt = self.wrt(indices=hrt_batch[:, 1]).view(-1, self.embedding_dim, self.relation_dim)
        return self.interaction_function(h=h, r=r, t=t, wh=wh, wt=wt).view(-1, 1)
                            
    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0]).unsqueeze(dim=1)
        r = self.relation_embeddings(indices=hr_batch[:, 1]).unsqueeze(dim=1)
        t = self.entity_embeddings(indices=None).unsqueeze(dim=0)
        wh = self.wrh(indices=hr_batch[:, 1]).view(-1, self.embedding_dim, self.relation_dim)
        wt = self.wrt(indices=hr_batch[:, 1]).view(-1, self.embedding_dim, self.relation_dim)
        return self.interaction_function(h=h, r=r, t=t, wh=wh, wt=wt)

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=None).unsqueeze(dim=0)
        r = self.relation_embeddings(indices=rt_batch[:, 0]).unsqueeze(dim=1)
        t = self.entity_embeddings(indices=rt_batch[:, 1]).unsqueeze(dim=1)
        wh = self.wrh(indices=hr_batch[:, 1]).view(-1, self.embedding_dim, self.relation_dim)
        wt = self.wrt(indices=hr_batch[:, 1]).view(-1, self.embedding_dim, self.relation_dim)
        return self.interaction_function(h=h, r=r, t=t, wh=wh, wt=wt)
