import jax.numpy as jnp

def getMapRange(positions, zaprange, probmap):
    """
    Fully vectorized function to extract probability maps for multiple units.
    
    Args:
        positions: jnp.array of shape (num_units, 2) representing unit positions.
        zaprange: Integer range of the zap attack.
        probmap: jnp.array of shape (map_size, map_size) representing the probability of enemy presence.

    Returns:
        filtered_maps: jnp.array of shape (num_units, submap_size, submap_size)
        x_lowers, y_lowers: jnp.array of shape (num_units,) containing the lower x and y indices.
    """
    num_units = positions.shape[0]
    map_size = probmap.shape[0]
    submap_size = 2 * zaprange + 1  # Size of extracted submap

    x_lowers = jnp.maximum(0, positions[:, 0] - zaprange)  
    x_uppers = jnp.minimum(map_size, positions[:, 0] + zaprange + 1)  
    y_lowers = jnp.maximum(0, positions[:, 1] - zaprange)  
    y_uppers = jnp.minimum(map_size, positions[:, 1] + zaprange + 1)  

    # Create index grids
    x_indices = jnp.arange(submap_size)[:, None] + x_lowers[:, None]  # (submap_size, num_units)
    y_indices = jnp.arange(submap_size)[None, :] + y_lowers[:, None]  # (num_units, submap_size)

    # Clip indices to ensure they stay within bounds
    x_indices = jnp.clip(x_indices, 0, map_size - 1)
    y_indices = jnp.clip(y_indices, 0, map_size - 1)

    # Gather the submaps for each unit
    filtered_maps = probmap[x_indices[:, :, None], y_indices[:, None, :]]

    return filtered_maps, x_lowers, y_lowers


def getZapCoords(positions, zaprange, probmap):
    """
    Fully vectorized function to get the best zap coordinates for multiple units.

    Args:
        positions: jnp.array of shape (num_units, 2), unit positions.
        zaprange: Integer range of the zap attack.
        probmap: jnp.array of shape (map_size, map_size), probability of enemy ships.

    Returns:
        zap_coords: jnp.array of shape (num_units, 2), best zap coordinates for each unit.
        zap_probs: jnp.array of shape (num_units,), probability values at those coordinates.
    """
    filtered_maps, x_lowers, y_lowers = getMapRange(positions, zaprange, probmap)

    # Find max probability position for each unit
    max_indices = jnp.argmax(filtered_maps.reshape(filtered_maps.shape[0], -1), axis=1)

    # Convert 1D indices to 2D (x, y) within local map
    local_xs, local_ys = jnp.divmod(max_indices, filtered_maps.shape[1])

    # Convert to global map coordinates
    global_xs = local_xs + x_lowers
    global_ys = local_ys + y_lowers

    # Get probability values at these coordinates
    zap_coords = jnp.stack([global_xs, global_ys], axis=1)
    zap_probs = probmap[global_xs, global_ys]

    return zap_coords, zap_probs


# === TEST CASE ===
if __name__ == "__main__":
    import numpy as np

    num_units = 16
    map_size = 24
    zap_range = 2

    # Random unit positions in range [0, map_size)
    unit_positions = jnp.array(np.random.randint(0, map_size, size=(num_units, 2)))

    # Create a random probability map
    probability_map = jnp.array(np.random.rand(map_size, map_size))

    # Run zap coordinate selection for all units
    zap_coords, zap_probs = getZapCoords(unit_positions, zap_range, probability_map)

    # Print results
    print("\nUnit Positions (x, y):")
    print(unit_positions)
    
    print("\nBest Zap Coordinates for Each Unit (x, y):")
    print(zap_coords)
    
    print("\nProbability Values at Selected Zap Coordinates:")
    print(zap_probs)
