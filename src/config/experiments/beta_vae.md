Basic Profile
```yaml
  beta_vae:
    # This profile is for the disentanglement experiment using a Beta-VAE.
    training:
      epochs: 150
      log_every_epoch: 25
    dirs:
      results_dir:
        side_by_side: "./side_by_side"
    optimizer:
      beta: 4.0
      params:
        lr: 0.001
    architecture:
      in_channels: 3
      latent_dim_digit: 20  # Dimensionality for the digit's latent space
      latent_dim_bbox: 10   # Dimensionality for the box's latent space
      use_resnet: false      # Use the more powerful decoder
```

Reconstruction 
```yaml
beta_vae_reconstruction:
    # This profile prioritizes high-quality reconstruction.
    # It uses a lower beta to reduce the pressure on the KL divergence term.
    training:
      epochs: 150
      log_every_epoch: 25
    dirs:
      results_dir:
        side_by_side: "./side_by_side"
    optimizer:
      beta: 1.0             # Low beta = focus on reconstruction
      params:
        lr: 0.001
    architecture:
      in_channels: 3
      latent_dim_digit: 32  # Give it enough capacity for shape
      latent_dim_bbox: 16   # Box is simple, needs less
      use_resnet: true      # Use the more powerful decoder
```

Strong displacement
```yaml
beta_vae_disentangled:
    # This profile prioritizes strong disentanglement.
    # It uses a higher beta and a larger latent space for the digit.
    training:
      epochs: 200
      log_every_epoch: 25
    dirs:
      results_dir:
      side_by_side: "./side_by_side"
    optimizer:
      beta: 5.0             # Higher beta = more pressure to disentangle
      params:
        lr: 0.001
    architecture:
      in_channels: 3
      latent_dim_digit: 64  # More capacity to capture digit style
      latent_dim_bbox: 16
      use_resnet: true      # Use the more powerful decoder
```