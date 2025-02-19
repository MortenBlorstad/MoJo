
# region imports

#  Core JAX & NumPy Imports
import jax
import jax.numpy as jnp
import numpy as np

#Flax (Neural Network Library for JAX)
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
# Optax (Optimization Library for JAX)
import optax

# logging at weighs and biases
import wandb

# endregion

# region Model Definition
class Encoder(nn.Module):
    latent_dim: int
    
    @nn.compact
    def __call__(self, x, rng = None, deterministic=False):
      
        x = nn.Conv(features=32, kernel_size=(3,3), strides=(2,2), padding='SAME')(x) # (batch, 12, 12, 32)
        skip2 = x  # Store for skip connection
        x = nn.silu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), strides=(2,2), padding='SAME')(x)  # (batch, 6, 6, 64)
        skip1 = x  # Store for skip connection
        x = nn.silu(x)
       
        x = x.reshape((x.shape[0], -1))  # Flatten
        mean = nn.Dense(self.latent_dim)(x)
        mean = nn.Dropout(rate=0.1)(mean, deterministic=deterministic, rng=rng)
        mean = nn.normalize(mean, axis=-1)
        log_var = nn.Dense(self.latent_dim)(x)
        return mean, log_var, skip1, skip2

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z, skip1, skip2):
        x = nn.Dense(6 * 6 * 64)(z)
        x = nn.silu(x)
        x = x.reshape((-1, 6, 6, 64))
        #x = x + skip1 # Skip connection
        x = nn.ConvTranspose(features=32, kernel_size=(3,3), strides=(2,2), padding='SAME')(x) # (batch, 6, 6, 64)
        #x = x + skip2 # Skip connection
        x = nn.silu(x)
        x = nn.ConvTranspose(features=16, kernel_size=(3,3), strides=(2,2), padding='SAME')(x) # (batch, 12, 12, 32)
        #x = nn.sigmoid(x)  # Output in range [0,1]
        x = nn.relu(x)
        return x

class Autoencoder(nn.Module):
    latent_dim: int
    
    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder()
    
    def reparameterize(self, mean, log_var, rng):
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng, shape=std.shape)
        return mean + eps * std
    
    def __call__(self, x, rng, deterministic=False):
        rng_dropout, rng_latent = jax.random.split(rng)
        mean, log_var, skip1, skip2 = self.encoder(x,rng_dropout,deterministic)
        z = self.reparameterize(mean, log_var, rng_latent)
        recon_x = self.decoder(z,skip1, skip2 )
        return mean,log_var, z, mean, recon_x 
# endregion




# ------------------ 1️⃣ Define Loss Function ------------------
def loss_fn(params, apply_fn, batch, rng):
    mu, log_var, latent, mu_reshaped, recon = apply_fn({"params": params}, batch, rng)
    mask =  jnp.where(batch > 0, 20, 1 )#1/9216)
    
    # Reconstruction loss (MSE)
    recon_loss = jnp.mean((mask*(recon - batch)) ** 2)
    sum_loss  = jnp.mean(((recon.sum(axis=-1) - batch.sum(axis=-1)))**2)
    

    # KL divergence loss
    kl_loss = -0.5 * jnp.mean(1 + log_var - jnp.square(mu) - jnp.exp(log_var))

    total_loss = recon_loss + 0.1 * kl_loss + sum_loss # Adjust KL weight
    

    return total_loss, (recon_loss, kl_loss, sum_loss)



# ------------------ 2️⃣ Define Train Step ------------------


@jax.jit
def train_step(state, batch, rng):
    """Compute gradients and return them instead of updating immediately."""
    rng, sub_rng = jax.random.split(rng)
    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, (recon_loss, kl_loss, BCE_loss)), grads = loss_grad_fn(state.params, state.apply_fn, batch, sub_rng)

    return grads, loss, (recon_loss, kl_loss, BCE_loss)






def plot_and_log_image(data, caption, vmin, vmax):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax, data in zip(axes.ravel(), data):
        im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)  # Control color range
    plt.colorbar(im, ax=axes, fraction=0.02, pad=0.04)  # Colorbar for reference
    # Convert to W&B Image
    wandb_image = wandb.Image(fig, caption=caption)
    plt.close()
    return wandb_image


if __name__ == "__main__":
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    latent_dim = 12 * 12  # Same as the output reshaped size
    
    pca = PCA(n_components=2)
    

    model = Autoencoder(latent_dim=latent_dim)
    # Example Training Loop
    rng = jax.random.PRNGKey(0)
    init_data = jnp.ones((32, 24, 24, 16))  # Dummy data for initialization
    variables = model.init(rng, init_data, rng)  # Initialize model with dummy input
    params = variables["params"]
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    data_dict  = np.load("MoJo/models/autoencoder/energy_map_dataset.npz")
   

    data = jnp.array(data_dict["arr_0"])
    num_epochs = 200
    batch_size = 32
    num_samples = data.shape[0]
    assert num_samples >= batch_size, "Batch size is larger than dataset!"

    # Initialize Weights & Biases project
    wandb.init(
    # set the wandb project where this run will be logged
    project="Autoencoder",
    # track hyperparameters and run metadata
    config={"learning_rate": 1e-6,
            "batch_size": batch_size}  # Log hyperparameters
    )
    
    n_iter = 1000
    loss_array  = np.zeros(n_iter)
    num_batches = data.shape[0] // batch_size   
    
    batched_train_step = jax.vmap(train_step, in_axes=(None, 0, None), out_axes=(None, 0, (0, 0, 0)))

    import functools
    @functools.partial(jax.jit, static_argnames=("num_batches",))
    def scan_train(state, batches, rng, num_batches):
        """Process multiple batches, average gradients, and update the model once."""
        
        def scan_fn(carry, batch_rng):
            state, grad_accum = carry
            batch, rng = batch_rng

            # Compute gradients for the batch
            grads, loss, losses = train_step(state, batch, rng)

            # Accumulate gradients
            new_grad_accum = jax.tree_util.tree_map(lambda g_acc, g: g_acc + g, grad_accum, grads)

            return (state, new_grad_accum), (loss, losses)

        # Initialize accumulated gradients as zero
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)

        # Perform scan over batches
        (state, accumulated_grads), (losses, loss_components) = jax.lax.scan(
            scan_fn, (state, zero_grads), (batches, jax.random.split(rng, num_batches))
        )

        # Average gradients
        averaged_grads = jax.tree_util.tree_map(lambda g: g / num_batches, accumulated_grads)

        # Apply the averaged gradients
        state = state.apply_gradients(grads=averaged_grads)

        return state, losses.mean(), (loss_components[0].mean(), loss_components[1].mean(), loss_components[2].mean())



    parallell_batches = 8
    for epoch in range(num_epochs):  # Example training loop
        rng, sub_rng = jax.random.split(rng)  # Ensure consistent randomness per step
        #perm = jax.random.permutation(sub_rng, num_samples) 
        perm = jax.random.choice(sub_rng, num_samples, shape=(parallell_batches * batch_size,), replace=False)
    
        shuffled_data = data[perm].reshape(parallell_batches, batch_size, *data.shape[1:])

        if epoch % 2 == 0:
            shuffled_data = jnp.transpose(shuffled_data, (0, 1, 3, 2, 4))  # Ensure correct transpose

        epoch_loss = 0.0
        
        
        state, loss, (recon_loss, kl_loss, BCE_loss) = scan_train(state, shuffled_data, sub_rng, num_batches=parallell_batches)


           
        avg_epoch_loss = loss.mean()

        if epoch % 2 == 0:
           
            # Convert JAX arrays to NumPy
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.6f}")

            wandb.log({
                "loss": avg_epoch_loss.item(),
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item(),
                "BCE_loss": BCE_loss.item()
            })

            if epoch % 10 == 0:
                variables = {"params": state.params}  # Wrap params properly
                batch = shuffled_data[0,:batch_size]  # Take a batch for visualization
                mu, log_var, latent, mu_reshaped, reconstructed = state.apply_fn(variables, batch, sub_rng, deterministic=True)
                mu_reshaped_np = np.array(mu_reshaped)  # (batch, 24, 24)
                reconstructed_np = np.array(reconstructed)  # (batch, 24, 24, 16)
                batch_np = np.array(batch)
                # Create and Log the plot
                embeddings_2d_pca = pca.fit_transform(mu_reshaped_np)
                plt.figure(figsize=(10, 10), dpi=100)
                plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], alpha=0.7)
                plt.title("2D Visualization of 32D Embeddings (PCA)")
                plt.xlabel("PCA Component 1")
                plt.ylabel("PCA Component 2")
                plt.grid(True)
                plt.tight_layout()
                wandb.log({
                    "latent": wandb.Image(plt),
                    "latent_img": plot_and_log_image(mu_reshaped_np.reshape( (batch_size, int(latent_dim**0.5), int(latent_dim**0.5))), caption=f"latent map Step {epoch}", vmin=0, vmax=1),
                    "sum energy":              plot_and_log_image(batch_np.sum(axis=-1), caption=f"sum map Step {epoch}", vmin=0, vmax =1),
                    "sum reconstructed energy": plot_and_log_image(reconstructed_np.sum(axis=-1),caption = f"sum rec map Step {epoch}", vmin=0, vmax =1),
                })
         
           
            
    wandb.finish()