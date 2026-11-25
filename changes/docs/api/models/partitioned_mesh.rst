Partitioned mesh displacement models
====================================

These experimental models learn spatial representations of fields on an
unstructured mesh. The mesh is split into disjoint core nodes. Each core
has a surrounding halo for message passing and can exchange information
with neighboring regions before decoding.

The autoencoder reconstructs the field at an observed time. The one-step
predictor consumes latent states for two successive observations and the
physical time increments. The predicted latent state is decoded using the
same autoencoder. For displacement data the three output channels can be
reported independently in the physical units of the dataset.

The temporal predictor runs a selective state-space model over a prescribed
time grid, starting from the encoded initial field. Train the temporal
predictor with a frozen spatial model in Stage 2. Stage 3 can tune the
spatial decoder and temporal predictor together.

.. autoclass:: physicsnemo.experimental.models.partitioned_mesh.PartitionAutoencoder
   :show-inheritance:

.. autoclass:: physicsnemo.experimental.models.partitioned_mesh.OneStepPredictor
   :show-inheritance:

.. autoclass:: physicsnemo.experimental.models.partitioned_mesh.TemporalPredictor
   :show-inheritance:

See the structural mechanics recipe in
``examples/structural_mechanics/partitioned_mesh_displacement`` for the graph
input contract, training and checkpoint export.
