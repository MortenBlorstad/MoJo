
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

# class Autoencoder(nn.Module):
#     latent_dim: int

#     @nn.compact
#     def __call__(self, x, rng):
#         # ===== Encoder: Downsample input to (batch, 24, 24 , 16) =====
#         x = nn.Conv(features=16, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24 , 16)
#         x = nn.silu(x)
#         x = nn.Conv(features=8, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24 , 8)
#         x = nn.silu(x)
#         x = nn.Conv(features=4, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24 , 4)
#         x = nn.silu(x)
#         x = nn.Conv(features=2, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24 , 1)
#         x = nn.silu(x)

#         # Flatten and project to (batch, 24, 24) latent space
#         x = x.reshape(x.shape[0], -1)  # (batch, 2*3*128)
#         mu = nn.Dense(features=self.latent_dim)(x)  # Mean (μ)
#         mu_norm  = jnp.linalg.norm(mu, axis=-1, keepdims=True) 
#         mu = mu/(mu_norm + 1e-8)
#         log_var = nn.Dense(features=self.latent_dim)(x)  # Log Variance (logσ²)

#         #mu_reshaped = mu.reshape(x.shape[0], 24, 24)
        
#         # ===== Reparameterization Trick (z = μ + σ * ε) =====
#         log_var = jnp.clip(log_var, -10, 10)
#         std = jnp.exp(0.5 * log_var)  # Compute standard deviation (σ)
#         epsilon = jax.random.normal(rng, std.shape)  # Sample ε ~ N(0,1)
#         latent = mu + std * epsilon  # Compute latent vector z
        
        


#         # ===== Decoder: Upsample back to (batch, 24, 24, 16) =====
#         x = nn.Dense(features=2 * 24 * 24)(latent)
#         x = x.reshape(x.shape[0], 24, 24, 2)  

#         x = nn.ConvTranspose(features=4, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x) 
#         x = nn.relu(x)

#         x = nn.ConvTranspose(features=8, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
#         x = nn.relu(x)

#         x = nn.ConvTranspose(features=16, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x) 
#         x = nn.sigmoid(x)  # Normalize output (assuming input is normali

#         return mu, log_var, latent, mu, x  # Return both latent space and reconstructed output


# class Autoencoder(nn.Module):
    # latent_dim: int
    # @nn.compact
    # def __call__(self, x, rng):
    #     # ===== Encoder: Reduce Channels, Preserve Spatial Size (24x24) input(batch, 24, 24, 16) =====
    #     x1 = nn.Conv(features=8, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24 , 32)
    #     x1 = nn.avg_pool(x1, window_shape=(1, 1), strides=(1, 1), padding="SAME")          #  (batch, 24, 24 , 32)
    #     x1 = nn.silu(x1)

    #     x2 = nn.Conv(features=4, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24 , 4)
    #     x2 = nn.max_pool(x2, window_shape=(3, 3), strides=(1, 1), padding="SAME")          #  (batch, 24, 24 , 4)
    #     x2 = nn.silu(x2)

    #     x3 = nn.Conv(features=2, kernel_size=(5, 5), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24 , 2)
    #     x3 = nn.max_pool(x3, window_shape=(5, 5), strides=(1, 1), padding="SAME")          #  (batch, 24, 24 , 2)
    #     x3 = nn.silu(x3)  # Fixed to use x3 instead of x2
    #     x = jnp.concatenate([x1, x2, x3], axis=-1)  # Concatenate along channel axis

    #     x4 = nn.Conv(features=2, kernel_size=(6, 6), strides=(1, 1), padding="SAME")(x)   # (batch, 24, 24 , 2)
    #     x4 = nn.avg_pool(x4, window_shape=(6, 6), strides=(1, 1), padding="SAME")         # (batch, 24, 24 , 2)      
    #     x4 = nn.silu(x4)  # Fixed to use x4 instead of x3

    #     x5 = nn.Conv(features=8, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24 , 8)
    #     x5 = nn.max_pool(x5, window_shape=(3, 3), strides=(2, 2))          #  (batch, 12, 12 , 8)
    #     x5 = nn.silu(x5)

    #     x5 = nn.Conv(features=6, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x5)  # (batch, 24, 24 , 16)
    #     x5 = nn.max_pool(x5, window_shape=(3, 3), strides=(2, 2))          #  (batch, 6, 6 , 16)
    #     x5 = nn.silu(x5)
    #     x5 = x5.reshape(x.shape[0], -1)  # Reshape to 1D
    #     x5 = nn.Dense(features=6 * 6 * 16)(x5)
    #     x5 = x5.reshape(x.shape[0], 24, 24, 1)  # Reshape to 2D

    #     x = jnp.concatenate([x1, x2, x3, x4, x5], axis=-1)  # Concatenate along channel axis
    #     skip2 = x
        
    #     x21 = nn.Conv(features=4, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(x)  #  (batch, 24, 24 , 4  )
    #     x21 = nn.avg_pool(x21, window_shape=(1, 1), strides=(1, 1), padding="SAME")          #  (batch, 24 , 24 , 4)
    #     x21 = nn.silu(x21)
        
    #     x22 = nn.Conv(features=2, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)   #(batch, 24, 24 , 2)
    #     x22 = nn.avg_pool(x22, window_shape=(3, 3), strides=(1, 1), padding="SAME")          #  (batch, 24, 24 , 2)
    #     x22 = nn.silu(x22)  # Final reduced channel output
    
    #     x23 = nn.Conv(features=2, kernel_size=(5, 5), strides=(1, 1), padding="SAME")(x)   #(batch, 24, 24 , 2)
    #     x23 = nn.avg_pool(x23, window_shape=(5, 5), strides=(1, 1), padding="SAME")          #  (batch, 24, 24 , 2)
    #     x23 = nn.silu(x23)  # Final reduced channel output

    #     x = jnp.concatenate([x21, x22, x23], axis=-1)                                       #(batch, 24, 24, 8)
    #     skip1 = x  # Skip connection for later
        
    #     mu = nn.Conv(features=1, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(x)   #(batch, 24, 24, 2)
    #     log_var = nn.Dense(24*24*1)(x.reshape(x.shape[0],-1))
    #     #
    #     # ===== VAE Reparameterization Trick =====
    #     #mu, log_var = jnp.split(x, 2, axis=-1)  # Split channels: First half = mu, Second half = log_var
    #     #mu = nn.tanh(mu)  # Normalize mu to [-1, 1]
        
    #     log_var = jnp.clip(log_var, -10, 10)  # Clip for numerical stability
    #     log_var = log_var.reshape(x.shape[0], 24,24, 1)
    #     std = jnp.exp(0.5 * log_var)  # Compute standard deviation (σ)

    #     epsilon = jax.random.normal(rng, std.shape)  # Sample noise from N(0,1)
    #     latent = mu + std * epsilon  # Reparameterized latent space (batch, 24, 24, 1)
        
    #     #jax.debug.print("Latent shape: {}", latent.shape)
    #     # first decoder step
    #     x1 = nn.Conv(features=4, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(latent)  # (batch, 24, 24, 4)
    #     x1 = nn.silu(x1)
    #     x2 = nn.Conv(features=2, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(latent)  # (batch, 24, 24, 4)
    #     x2 = nn.silu(x2)
    #     x3 = nn.Conv(features=2, kernel_size=(5, 5), strides=(1, 1), padding="SAME")(latent)  # (batch, 24, 24, 4)
    #     x3 = nn.silu(x3)
    #     x = jnp.concatenate([x1, x2, x3], axis=-1)     # (batch, 24, 24, 8)

    #     x = x + skip1

    #     # second decoder step
    #     x4 = nn.Conv(features=8, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24, 2)
    #     x4 = nn.silu(x4)
    #     x5 = nn.Conv(features=4, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24, 2)
    #     x5 = nn.silu(x5)
        
    #     x6 = nn.Conv(features=2, kernel_size=(5, 5), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24, 2)
    #     x6 = nn.silu(x6)

    #     x7 = nn.Conv(features=2, kernel_size=(6, 6), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24, 2)
    #     x7 = nn.silu(x7)    
        
    #     x = jnp.concatenate([x4, x5, x6, x7], axis=-1)     # (batch, 24, 24, 16)
    #     x = x + skip2
    #     x = nn.Conv(features=16, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(x)  # (batch, 24, 24, 16)
    #     #jax.debug.print("x shape: {}", x.shape)
    #     x = nn.sigmoid(x) 
    #     return mu, log_var, latent, mu, x  # Return both latent space and reconstructed output

class Encoder(nn.Module):
    latent_dim: int
    
    @nn.compact
    def __call__(self, x):
        x1 = nn.Conv(features=4, kernel_size=(1,1), strides=(1,1), padding='SAME')(x) # (batch, 24, 24, 4)
        x1 = nn.avg_pool(x1, window_shape=(1,1), strides=(1,1), padding='SAME')  # (batch, 24, 24, 4)
        
        x2 = nn.Conv(features=4, kernel_size=(3,3), strides=(1,1), padding='SAME')(x)  # (batch, 24, 24, 4)
        x = jnp.concatenate([x1, x2], axis=-1)  # Concatenate along channel axis
   
        skip1 = x  # Store for skip connection
        x = nn.relu(x)
       

        x1 = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(x)   #(batch, 24, 24, 2)
        x1 = nn.avg_pool(x1, window_shape=(1,1), strides=(1,1), padding='SAME')  # (batch, 24, 24, 4)

        x2 = nn.Conv(features=2, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)   #(batch, 24, 24, 2)
        x = jnp.concatenate([x1, x2], axis=-1)  # Concatenate along channel axis
        skip2 = x  # Store for skip connection
        x = nn.relu(x)

        x = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(x)
        
        mu,log_var = jnp.split(x,indices_or_sections=2, axis=-1)
        mu = nn.tanh(mu)

        return mu, log_var, skip1, skip2

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z, skip1, skip2):
        x =  nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(z)
        
        x1 = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(x)   #(batch, 24, 24, 2)
        x2 = nn.Conv(features=2, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)   #(batch, 24, 24, 2)
        x = jnp.concatenate([x1, x2], axis=-1)  # Concatenate along channel axis
        x = nn.relu(x)
        x = x + skip2

        x1 = nn.Conv(features=4, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(x)   #(batch, 24, 24, 2)
        x2 = nn.Conv(features=4, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)   #(batch, 24, 24, 2)
        x = jnp.concatenate([x1, x2], axis=-1)  # Concatenate along channel axis
        x = nn.relu(x)
        x = x + skip1

        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
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
    
    def __call__(self, x, rng):
        mean, log_var, skip1, skip2 = self.encoder(x)
        z = self.reparameterize(mean, log_var, rng)
        recon_x = self.decoder(z, skip1, skip2)
        return mean, log_var, z, mean, recon_x 

# ------------------ 1️⃣ Define Loss Function ------------------
def loss_fn(params, apply_fn, batch, rng):
    mu, log_var, latent, mu_reshaped, recon = apply_fn({"params": params}, batch, rng)
    mask =  jnp.where(batch > 0, 576, 1/9216)
    # Reconstruction loss (MSE)
    recon_loss = jnp.mean((mask * (recon - batch)) ** 2)
    sum_loss  = jnp.mean(((recon.sum(axis=-1) - batch.sum(axis=-1)))**2)

    
    # KL divergence loss
    kl_loss = -0.5 * jnp.mean(1 + log_var - jnp.square(mu) - jnp.exp(log_var))

    total_loss = recon_loss + 0.1 * kl_loss + sum_loss # Adjust KL weight
    


    return total_loss, (recon_loss, kl_loss, sum_loss)



# ------------------ 2️⃣ Define Train Step ------------------
@jax.jit
def train_step(state, batch, rng):
    # Compute loss and gradients
    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    
    # Correctly pass apply_fn
    (loss, (recon_loss, kl_loss, BCE_loss)), grads = loss_grad_fn(state.params, state.apply_fn, batch, rng)
    
    # Update model parameters
    state = state.apply_gradients(grads=grads)
    
    return state, loss, (recon_loss, kl_loss, BCE_loss)
    


if __name__ == "__main__":
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    def plot_and_log_image(data, caption, vmin, vmax):
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)  # Control color range
        plt.colorbar(im, ax=ax)  # Optional: Add colorbar for reference
        plt.title(caption)
        
        # Convert to W&B Image
        wandb_image = wandb.Image(fig, caption=caption)
        plt.close(fig)  # Close the figure to prevent duplication
        return wandb_image


    latent_dim = 3  # Same as the output reshaped size
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
    num_epochs = 100
    batch_size = 32
    num_samples = data.shape[0]
    assert num_samples >= batch_size, "Batch size is larger than dataset!"

    # Initialize Weights & Biases project
    wandb.init(
    # set the wandb project where this run will be logged
    project="Autoencoder",
    # track hyperparameters and run metadata
    config={"learning_rate": 1e-3,
            "batch_size": batch_size}  # Log hyperparameters
)
    
    n_iter =1000
    loss_array  = np.zeros(n_iter)
    num_batches = data.shape[0] // batch_size   

    for epoch in range(num_epochs):  # Example training loop
        rng, sub_rng = jax.random.split(rng)  # Ensure consistent randomness per step
        perm = jax.random.permutation(sub_rng, num_samples) 
        shuffled_data = data[perm]
        epoch_loss = 0.0
        
        for i in range(num_batches):
            batch = shuffled_data[i * batch_size:(i + 1) * batch_size]
            if epoch % 2 ==0:
                batch = jnp.transpose(batch, (0, 2, 1, 3))
            rng, sub_rng = jax.random.split(rng)
            state, loss, (recon_loss, kl_loss, BCE_loss) = train_step(state, batch, sub_rng)
            epoch_loss += loss
           
            wandb.log({"step": epoch * num_batches + i,
                        "loss": loss.item(),
                        "recon_loss": recon_loss.item(),
                        "kl_loss": kl_loss.item(),
                        "bce_loss": BCE_loss.item() }) 
        avg_epoch_loss = epoch_loss / num_batches

        if epoch % 2 == 0:
            variables = {"params": state.params}  # Wrap params properly
            mu, log_var, latent, mu_reshaped, reconstructed = state.apply_fn(variables, batch, sub_rng)

            # Convert JAX arrays to NumPy
            mu_reshaped_np = np.array(mu_reshaped)  # (batch, 24, 24)
            reconstructed_np = np.array(reconstructed)  # (batch, 16, 24, 24)
            batch_np = np.array(batch)
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.6f}")
      
            

            if epoch % 10 == 0:

                mu_reshaped_np = np.array(mu_reshaped)  # (batch, 24, 24)
                reconstructed_np = np.array(reconstructed)  # (batch, 24, 24, 16)
                batch_np = np.array(batch)
                # Create and Log the plot
                wandb.log({
                    "latent_img": plot_and_log_image(mu_reshaped_np[0], caption=f"latent map Step {epoch}", vmin=-1, vmax =1),
                    "sum energy":              plot_and_log_image(batch_np[0].sum(axis=-1), caption=f"sum map Step {epoch}", vmin=0, vmax =1),
                    "sum reconstructed energy": plot_and_log_image(reconstructed_np[0].sum(axis=-1),caption = f"sum rec map Step {epoch}", vmin=0, vmax =1),
                })
            

    wandb.finish()