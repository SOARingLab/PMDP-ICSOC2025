
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap
import os
import ast
from matplotlib.ticker import MaxNLocator
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
import matplotlib.scale as mscale
from matplotlib.colors import to_rgba, to_rgb

plot_counter = 0

################ The experiment for the tranining time (each-episode) in different candidate services and abstract services ################

def load_data_from_file(
    data_dir="",
    sub_dir="",
    filename=""
):
    """
    Load experimental data from a text file containing a nested Python list structure.
    
    Data Structure:
        The file should contain a 2D list (e.g., [[v11, v12, ...], [v21, v22, ...]]), where:
        - Each sublist represents a sequence of measurements (e.g., per episode)
        - Each element is a numerical value (float/int) representing metrics like:
          - Success rates
          - Rewards
          - Training time
          - Other experiment-specific metrics
    
    Parameters:
        data_dir (str): Name of the root data directory (default: "")
        sub_dir (str): Subdirectory under data_dir (default: "")
        filename (str): Name of the text file to load (default: "")
    
    Returns:
        list: Parsed 2D list of numerical values, or None if an error occurs.
    """

    # Validate input parameters (check for empty values)
    if not data_dir.strip():
        print("\n\n ########################## Error: data_dir cannot be empty. Please provide a valid directory name. ##########################\n\n")
        return None
    if not sub_dir.strip():
        print("\n\n ########################## Error: sub_dir cannot be empty. Please provide a valid subdirectory name. ##########################\n\n")
        return None
    if not filename.strip():
        print("\n\n ########################## Error: filename cannot be empty. Please provide a valid filename. ##########################\n\n")
        return None
    

    # 1. Get the directory where the current script is located (experiments_visualizer directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current script directory:", current_dir)
    
    # 2. Calculate the project root directory (go up two levels from current directory)
    project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
    print("Project root directory:", project_root)
    
    # 3. Construct the data file path using function parameters
    file_path = os.path.join(project_root, data_dir, sub_dir, filename)
    file_path = os.path.normpath(file_path)  # Normalize the path (handle slashes)
    print("Data file path:", file_path)
    
    # 4. Read and parse the data
    try:
        with open(file_path, 'r') as f:
            data_str = f.read()
            data = ast.literal_eval(data_str)
        print(f"Data loaded successfully, total {len(data)} rows")
        return data
    except FileNotFoundError:
        print(f"\n\n ########################## Error: File not found at {file_path}. Please check the path components. ##########################\n\n")
        return None
    except Exception as e:
        print(f"\n\n ########################## Failed to load data: {e} ##########################\n\n")
        return None



def plot_Experiment_pictures(
    instance_evaluation_rewards_2Dlist, 
    labels, 
    title='Average Evaluation Reward over Episodes',
    xlabel='Episodes (×10³)',  
    ylabel='Average Reward', 
    colors=None, 
    markers=None, 
    eval_interval=5,
    window_size=1,           
    trend_window_size=5,     
    zero_start=False,
    grid_on=False,
    inset_mode='below',  # New options: 'inside', 'below', 'main', or 'sub'
    y_compress_main=None,     
    y_compress_inset=None,    
    scale_ratio_main=0.5,     
    scale_ratio_inset=0.5,    
    fontsize_increase=5,
    color_group_size=1,  
    plot_mode='line',  
    show_error_shadow=True,  
    x_axis_start=0,  
    line_type='Solid_and_dotted',  
    label_mode='auto',
    zoom_enabled=False,      # New parameter: whether to enable zoomed subplot
    zoom_range=None,          # New parameter: x-axis range to zoom in, e.g., (0.4, 0.8)
    zoom_y_range=None,   # New parameter: y-axis range for the zoomed subplot
    x_range=None,  # New parameter: x-range for calculating AUC and average metrics, e.g., (0.5, 2.0) 
                    # Metrics are calculated between 0.5k and 2k episodes
    ax=None,  # New parameter: default None, adjusts plotting logic based on whether ax is provided
              # Used for plotting 2 subplots in one figure
    legend_position='upper left'
):
    """
    Visualizes experimental evaluation data (e.g., rewards or success rates) over episodes.
    
    Generates line/point plots with optional error shadows, trend subplots, zoomed regions,
    and calculates statistical metrics (AUC, average values) within specified ranges. 
    Supports flexible styling, axis configurations, and subplot layouts.
    
    Key Parameters:
        instance_evaluation_rewards_2Dlist: 2D list of evaluation data (rows = experiments, columns = episodes)
        labels: List of labels for each experiment curve
        plot_mode: 'line' or 'point' for different visualization styles
        x_range: Tuple (start, end) defining x-axis range for metric calculation (AUC, averages)
        zoom_enabled: If True, adds a zoomed inset subplot for detailed viewing
        ax: Optional matplotlib Axes object for external subplot integration
    
    Returns:
        list: Statistical results including AUC, average success rate, and standard error for each curve
    """

    # Tools for zoomed subplot
    if zoom_enabled:
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

    def lighten_color(color, amount=0.5):
        """Lighten the given color by a specified amount"""
        try:
            c = to_rgb(color)
        except ValueError:
            c = to_rgb('blue')
        return tuple([min(1, x + amount * (1 - x)) for x in c])

    class CompressYScale(ScaleBase):
        """Custom y-axis scale for compressing a specific y-range"""
        name = 'compressy'
        
        def __init__(self, axis, **kwargs):
            super().__init__(axis)
            self.y_start = kwargs.get('y_start', 0)
            self.y_end = kwargs.get('y_end', 1)
            self.scale_ratio = kwargs.get('scale_ratio', 0.5)
            
        def get_transform(self):
            return self.CompressYTransform(self.y_start, self.y_end, self.scale_ratio)
        
        def set_default_locators_and_formatters(self, axis):
            pass
        
        def limit_range_for_scale(self, vmin, vmax, minpos):
            return vmin, vmax
        
        class CompressYTransform(Transform):
            """Transform for compressing the specified y-range"""
            input_dims = 1
            output_dims = 1
            is_separable = True
            
            def __init__(self, y_start, y_end, scale_ratio):
                super().__init__()
                self.y_start = y_start
                self.y_end = y_end
                self.scale_ratio = scale_ratio
            
            def transform_non_affine(self, y):
                y = np.array(y)
                mask1 = y < self.y_start
                mask2 = (y >= self.y_start) & (y <= self.y_end)
                mask3 = y > self.y_end
                
                y_transformed = np.empty_like(y)
                y_transformed[mask1] = y[mask1]
                y_transformed[mask2] = self.y_start + (y[mask2] - self.y_start) * self.scale_ratio
                y_transformed[mask3] = self.y_start + (self.y_end - self.y_start) * self.scale_ratio + (y[mask3] - self.y_end)
                
                return y_transformed
            
            def inverted(self):
                return CompressYScale.InvertedCompressYTransform(self.y_start, self.y_end, self.scale_ratio)
        
        class InvertedCompressYTransform(Transform):
            """Inverse transform for decompressing the specified y-range"""
            input_dims = 1
            output_dims = 1
            is_separable = True
            
            def __init__(self, y_start, y_end, scale_ratio):
                super().__init__()
                self.y_start = y_start
                self.y_end = y_end
                self.scale_ratio = scale_ratio
            
            def transform_non_affine(self, y):
                y = np.array(y)
                y_transformed = np.empty_like(y)
                
                mask1 = y < self.y_start
                mask2 = (y >= self.y_start) & (y <= self.y_start + (self.y_end - self.y_start) * self.scale_ratio)
                mask3 = y > self.y_start + (self.y_end - self.y_start) * self.scale_ratio
                
                y_transformed[mask1] = y[mask1]
                y_transformed[mask2] = self.y_start + (y[mask2] - self.y_start) / self.scale_ratio
                y_transformed[mask3] = self.y_end + (y[mask3] - self.y_start - (self.y_end - self.y_start) * self.scale_ratio)
                
                return y_transformed
            
            def inverted(self):
                return CompressYScale.CompressYTransform(self.y_start, self.y_end, self.scale_ratio)

    mscale.register_scale(CompressYScale)

    # Initialize list for statistical results: AUC, mean rewards, and standard errors
    stats = []

    if len(instance_evaluation_rewards_2Dlist) != len(labels):
        raise ValueError("instance_evaluation_rewards_2Dlist and labels must have the same length.")


    num_curves = len(instance_evaluation_rewards_2Dlist)
    
    # Color and marker processing
    if colors is None:
        base_colors = plt.cm.tab10.colors
        if color_group_size > 1:
            colors = [base_colors[i // color_group_size % len(base_colors)] for i in range(num_curves)]
        else:
            colors = list(base_colors)
            if num_curves > len(colors):
                colors = colors * (num_curves // len(colors) + 1)
            colors = colors[:num_curves]
    else:
        if len(colors) < num_curves:
            colors = colors * (num_curves // len(colors) + 1)
        colors = colors[:num_curves]
        
    if markers is None:
        markers = ['o', '^', 's', 'D', 'v', '>', '<', 'p', '*', 'h']
        if num_curves > len(markers):
            markers = markers * (num_curves // len(markers) + 1)
    
    if plot_mode == 'line':
        if line_type == 'Solid_and_dotted':
            line_styles = ['-' if i % 2 == 0 else '--' for i in range(num_curves)]
            if num_curves > len(line_styles):
                line_styles *= (num_curves // len(line_styles) + 1)
        elif line_type == 'auto':
            all_styles = ['--', '-.', '-', ':', (0, (3, 1, 1, 1))]
            line_styles = [all_styles[i % len(all_styles)] for i in range(num_curves)]
        else:
            line_styles = ['-' if i % 2 == 0 else '--' for i in range(num_curves)]
            if num_curves > len(line_styles):
                line_styles *= (num_curves // len(line_styles) + 1)
    else:
        line_styles = [None] * num_curves

    # Adjust plotting logic based on whether ax is provided
    if ax is None:
        # Create main plot and subplot based on inset_mode
        fig = plt.figure(figsize=(12, 8))
        if inset_mode == 'main':
            ax_main = fig.add_subplot(111)
            ax_inset = None
        elif inset_mode == 'sub':
            ax_main = None
            ax_inset = fig.add_subplot(111)
        elif inset_mode == 'inside':
            ax_main = fig.add_subplot(111)
            if 'CSSC' in labels[1] and '50' in labels[1]:
                ax_inset = ax_main.inset_axes([0.10, 0.48, 0.45, 0.45])
            else:
                ax_inset = ax_main.inset_axes([0.50, 0.05, 0.45, 0.45])
        elif inset_mode == 'below':
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
            ax_main = fig.add_subplot(gs[0])
            ax_inset = fig.add_subplot(gs[1], sharex=ax_main)
        else:
            raise ValueError("inset_mode must be 'inside', 'below', 'main', or 'sub'")
    else:
        # Use provided Axes, disable subplot creation
        ax_main = ax
        ax_inset = None
        fig = ax.figure

    # Plot main and inset figures
    for idx, (experiment, label) in enumerate(zip(instance_evaluation_rewards_2Dlist, labels)):
        experiment = np.array(experiment)
        if experiment.ndim != 2:
            raise ValueError(f"Data for experiment {idx} must be 2-dimensional.")
        
        mean_rewards = np.mean(experiment, axis=0)
        std_rewards = np.std(experiment, axis=0)
        se_rewards = std_rewards / np.sqrt(experiment.shape[0])  # Standard error
        
        # Smoothing for error shadow
        window_size_error = max(1, window_size)
        if window_size_error > 1 and len(mean_rewards) >= window_size_error:
            kernel_error = np.ones(window_size_error) / window_size_error
            mean_padded = np.pad(mean_rewards, (window_size_error - 1, 0), mode='edge')
            mean_rewards_smoothed = np.convolve(mean_padded, kernel_error, mode='valid')
            se_padded = np.pad(se_rewards, (window_size_error - 1, 0), mode='edge')
            se_rewards_smoothed = np.convolve(se_padded, kernel_error, mode='valid')
        else:
            mean_rewards_smoothed = mean_rewards
            se_rewards_smoothed = se_rewards
        
        # Smoothing for trend curve
        window_size_trend = max(1, trend_window_size)
        if window_size_trend > 1 and len(mean_rewards) >= window_size_trend:
            kernel_trend = np.ones(window_size_trend) / window_size_trend
            mean_padded_trend = np.pad(mean_rewards, (window_size_trend - 1, 0), mode='edge')
            mean_trend_smoothed = np.convolve(mean_padded_trend, kernel_trend, mode='valid')
        else:
            mean_trend_smoothed = mean_rewards

        # Generate x-axis (episodes)
        if not zero_start:
            episodes = np.arange(1, len(mean_rewards_smoothed) + 1) * eval_interval
            episodes_trend = np.arange(1, len(mean_trend_smoothed) + 1) * eval_interval
        else:
            episodes = np.arange(0, len(mean_rewards_smoothed)) * eval_interval
            episodes_trend = np.arange(0, len(mean_trend_smoothed)) * eval_interval
        
        episodes += x_axis_start
        episodes_trend += x_axis_start
        episodes_scaled = episodes / 1000.0  # Convert to thousands of episodes
        episodes_trend_scaled = episodes_trend / 1000.0

        # Set line/marker styles based on plot_mode
        if plot_mode == 'point':
            main_marker = 's' if idx % 2 == 0 else '^'
            inset_marker = main_marker
            main_linestyle = 'None'
            inset_linestyle = 'None'
        else:
            main_marker = None
            inset_marker = None
            main_linestyle = line_styles[idx]
            inset_linestyle = line_styles[idx]

        # Plot on main axis
        if ax_main is not None:
            ax_main.plot(
                episodes_scaled,
                mean_rewards_smoothed,
                color=colors[idx],
                linestyle=main_linestyle,
                marker=main_marker,
                label=label,
                linewidth=2
            )
            if show_error_shadow:
                shadow_color = colors[idx] if idx % 2 == 0 else lighten_color(colors[idx], 0.5)
                ax_main.fill_between(
                    episodes_scaled,
                    mean_rewards_smoothed - se_rewards_smoothed,
                    mean_rewards_smoothed + se_rewards_smoothed,
                    color=shadow_color,
                    alpha=0.2
                )
        
        # Plot on inset axis
        if ax_inset is not None:
            ax_inset.plot(
                episodes_trend_scaled,
                mean_trend_smoothed,
                color=colors[idx],
                linestyle=inset_linestyle,
                marker=inset_marker,
                linewidth=2
            )


        # Process x_range to calculate AUC and average metrics
        episodes_scaled = episodes / 1000.0  # Original x-axis data
        if x_range is not None:
            if len(x_range) != 2:
                raise ValueError("x_range must be a tuple of two elements (start, end).")
            x_start, x_end = x_range
            mask = (episodes_scaled >= x_start) & (episodes_scaled <= x_end)
            
            if np.any(mask):
                selected_mean = mean_rewards_smoothed[mask]
                selected_se = se_rewards_smoothed[mask]
                selected_x = episodes_scaled[mask]

                # Calculate AUC (Area Under Curve)
                auc = np.trapz(selected_mean, selected_x)

                # Calculate average mean and average standard error
                avg_mean = np.mean(selected_mean)
                avg_se_mean = np.mean(selected_se)  # Standard error of the mean

                stats.append({
                    'label': label,
                    'auc': auc,
                    'avg_SR': avg_mean,  # Average success rate/mean
                    'avg_SE': avg_se_mean  # Average standard error
                })
            else:
                stats.append({
                    'label': label,
                    'auc': None,
                    'avg_SR': None,
                    'avg_SE': None
                })


    # Apply y-axis compression configurations
    if ax_main is not None and y_compress_main is not None:
        y_start_main, y_end_main = y_compress_main
        ax_main.set_yscale('compressy', y_start=y_start_main, y_end=y_end_main, scale_ratio=scale_ratio_main)
    if ax_inset is not None and y_compress_inset is not None:
        y_start_inset, y_end_inset = y_compress_inset
        ax_inset.set_yscale('compressy', y_start=y_start_inset, y_end=y_end_inset, scale_ratio=scale_ratio_inset)
    
    # Configure main axis
    if ax_main is not None:
        ax_main.set_ylabel(ylabel, fontsize=14 + fontsize_increase)
        ax_main.grid(grid_on)
        handles, legend_labels = ax_main.get_legend_handles_labels()
        
        if label_mode == 'manual':
            if 'CSSC' in labels[1] and '50' in labels[1]:
                ax_main.legend(handles, legend_labels, loc='upper right', fontsize=8 + fontsize_increase, ncol=3)
            elif ('ω' in labels[0] or 'α' in labels[0]):
                ### For Fig.6a-b legends
                ax_main.legend(handles, legend_labels, loc='lower left', fontsize=11 + fontsize_increase, ncol=1)  
            else:
                ax_main.legend(handles, legend_labels, loc='upper left', fontsize=8 + fontsize_increase, ncol=3)                    
        else:
            ax_main.legend(handles, legend_labels, loc=legend_position, fontsize=8 + fontsize_increase, ncol=5)
        
        ax_main.tick_params(axis='both', which='major', labelsize=12 + fontsize_increase)
    
    # Configure inset axis
    if ax_inset is not None:
        ax_inset.tick_params(axis='both', which='major', labelsize=12 + fontsize_increase)
        ax_inset.grid(True)
        ax_inset.set_ylabel('')
    
    # Configure x-axis labels based on inset mode
    if inset_mode == 'below' and ax_inset is not None and ax_main is not None:
        ax_inset.set_xlabel(xlabel, fontsize=14 + fontsize_increase)
        ax_main.set_xlim(ax_main.get_xlim())
    elif ax_main is not None and inset_mode != 'sub':
        ax_main.set_xlabel(xlabel, fontsize=14 + fontsize_increase)
        ax_main.xaxis.set_major_locator(MaxNLocator(nbins=11))
    if ax_inset is not None:
        ax_inset.xaxis.set_major_locator(MaxNLocator(nbins=11))
    
    # Add title to inset axis
    if inset_mode == 'inside' and ax_inset is not None:
        ax_inset.text(
            0.5, 1.05,
            f'Trend Curves with {trend_window_size}-Pt Moving Avg. Smoothing',
            horizontalalignment='center',
            fontsize=12 + fontsize_increase - 1,
            transform=ax_inset.transAxes,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2)
        )
    elif ax_inset is not None and inset_mode not in ['main', 'sub']:
        ax_inset.set_title(f'Trend Curves with {trend_window_size}-Pt Moving Avg. Smoothing', fontsize=12 + fontsize_increase)

    # Add zoomed subplot if enabled
    if zoom_enabled and zoom_range is not None and ax_main is not None:
        # Create inset axis for zoomed region (bottom-right of main axis)
        ax_zoom = ax_main.inset_axes([0.38, 0.05, 0.6, 0.55])  # [x0, y0, width, height] in main axis coordinates

        # Set font size for zoomed subplot
        ax_zoom.tick_params(axis='x', labelsize=16)
        ax_zoom.tick_params(axis='y', labelsize=16)

        # Plot data in zoomed subplot
        for idx, experiment in enumerate(instance_evaluation_rewards_2Dlist):
            experiment = np.array(experiment)
            if experiment.ndim != 2:
                continue  # Skip invalid data

            # Calculate mean and standard error (same as main plot)
            mean_rewards = np.mean(experiment, axis=0)
            std_rewards = np.std(experiment, axis=0)
            se_rewards = std_rewards / np.sqrt(experiment.shape[0])
            
            # Smooth data for error shadow
            window_size_error = max(1, window_size)
            if window_size_error > 1 and len(mean_rewards) >= window_size_error:
                kernel_error = np.ones(window_size_error) / window_size_error
                mean_padded = np.pad(mean_rewards, (window_size_error - 1, 0), mode='edge')
                mean_rewards_smoothed = np.convolve(mean_padded, kernel_error, mode='valid')
                se_padded = np.pad(se_rewards, (window_size_error - 1, 0), mode='edge')
                se_rewards_smoothed = np.convolve(se_padded, kernel_error, mode='valid')
            else:
                mean_rewards_smoothed = mean_rewards
                se_rewards_smoothed = se_rewards

            # Generate x-axis data (same as main plot)
            if not zero_start:
                episodes = np.arange(1, len(mean_rewards_smoothed) + 1) * eval_interval
            else:
                episodes = np.arange(0, len(mean_rewards_smoothed)) * eval_interval
            episodes += x_axis_start
            episodes_scaled = episodes / 1000.0

            # Filter data within zoom range
            mask = (episodes_scaled >= zoom_range[0]) & (episodes_scaled <= zoom_range[1])
            if not np.any(mask):
                continue

            # Plot in zoomed subplot
            if plot_mode == 'point':
                marker = 's' if idx % 2 == 0 else '^'
                ax_zoom.plot(
                    episodes_scaled[mask],
                    mean_rewards_smoothed[mask],
                    color=colors[idx],
                    marker=marker,
                    linestyle='None',
                    linewidth=2
                )
            else:
                ax_zoom.plot(
                    episodes_scaled[mask],
                    mean_rewards_smoothed[mask],
                    color=colors[idx],
                    linestyle=line_styles[idx],
                    linewidth=2
                )
            
            # Add error shadow if enabled
            if show_error_shadow:
                shadow_color = colors[idx] if idx % 2 == 0 else lighten_color(colors[idx], 0.5)
                ax_zoom.fill_between(
                    episodes_scaled[mask],
                    mean_rewards_smoothed[mask] - se_rewards_smoothed[mask],
                    mean_rewards_smoothed[mask] + se_rewards_smoothed[mask],
                    color=shadow_color,
                    alpha=0.2
                )
        
        # Configure zoomed subplot ranges
        ax_zoom.set_xlim(zoom_range)
        if zoom_y_range is not None:
            ax_zoom.set_ylim(zoom_y_range)
        else:
            # Auto-determine y-range if not specified
            all_y = []
            for line in ax_zoom.get_lines():
                all_y.extend(line.get_ydata())
            if all_y:
                ax_zoom.set_ylim(min(all_y), max(all_y))
        
        ax_zoom.grid(True)
        # Mark zoomed region on main axis
        mark_inset(ax_main, ax_zoom, loc1=4, loc2=3, fc="none", ec="0.5")

    # Set title and labels
    ax_main.set_title(title, fontsize=13 + fontsize_increase)
    ax_main.set_xlabel(xlabel, fontsize=13 + fontsize_increase)
    ax_main.set_ylabel(ylabel, fontsize=13 + fontsize_increase)

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    
    
    
    
    if os.environ.get('IS_IN_DOCKER') == 'true' and ax is None:
        # for docker env, save picture only
        
        global plot_counter
        plot_counter += 1
        filename = f"plot_result_{plot_counter}.png" # e.g.,: plot_result_1.png, plot_result_2.png

        output_path = os.path.join("/app/src/training_records", filename)

        plt.savefig(output_path)
        print(f"Running in Docker. Plot saved to {output_path}")
    else:
        # local windows, show directly
        print("Running in local environment. Displaying plot.")
        
        # Show plot if no external ax is provided
        if ax is None:
            plt.show()


    # Show plot if no external ax is provided
    #if ax is None:
        #plt.show()

    return stats  # Return statistical results



def run_comparison_experiment(
    # P-MDP data file path parameters (10AS, 30AS, 50AS)
    pmdp_10as_path,
    pmdp_30as_path,
    pmdp_50as_path,
    # CSSC-MDP data file path parameters (10AS, 30AS, 50AS)
    cssc_10as_5w_path,
    cssc_30as_2w_path,
    cssc_50as_2w_path,
    # Data processing parameters (truncation range)
    cssc_5w_to_2w_cut=100,  # Take first 100 elements from 5w data as 2w data
    cssc_final_cut_start=2,  # Final data starts truncating from index 2
    cssc_final_cut_end=100,  # Final data truncates up to index 100
    # Plotting parameters
    title='',
    xlabel='Thousands of episodes',
    ylabel='Service Composition Success Rate',
    eval_interval=200,
    window_size_point=1,
    window_size_line=5,
    trend_window_size=5,
    inset_mode='main',
    y_compress_main=(0.0, 0.0),
    y_compress_inset=(0.0, 0.0),
    color_group_size=2,
    grid_on_point=False,
    grid_on_line=True,
    show_error_shadow_point=True,
    show_error_shadow_line=False,
    x_axis_start=400,
    label_mode='manual',
    fontsize_increase=10
):
    """
    Run comparison experiment between CSSC-MDP and P-MDP, including data loading, processing and plotting.
    
    Parameters:
        pmdp_10as_path (dict): Path params for Ex1_PMDP_10AS_WS1_SuccessRate (data_dir, sub_dir, filename)
        pmdp_30as_path (dict): Path params for Ex1_PMDP_30AS_WS1_SuccessRate
        pmdp_50as_path (dict): Path params for Ex1_PMDP_50AS_WS1_SuccessRate
        cssc_10as_5w_path (dict): Path params for CSSC_10AS_WS1_SuccessRate_200Interval_5wEpisode
        cssc_30as_2w_path (dict): Path params for CSSC_30AS_WS1_SuccessRate_200Interval_2wEpisode
        cssc_50as_2w_path (dict): Path params for CSSC_50AS_WS1_SuccessRate_200Interval_2wEpisode
        cssc_5w_to_2w_cut (int): Number of elements to keep when converting 5w data to 2w data
        cssc_final_cut_start (int): Start index for final data cutting
        cssc_final_cut_end (int): End index for final data cutting
        Other params: Plotting configurations for plot_Experiment_pictures
    
    Returns:
        None
    """
    # -------------------------- 1. Load all data --------------------------
    def load_data(path):
        """Helper function: Load data via path parameters"""
        data = load_data_from_file(
            data_dir=path['data_dir'],
            sub_dir=path['sub_dir'],
            filename=path['filename']
        )
        if data is None:
            print(f"Failed to load data from {path}, experiment aborted.")
        return data

    # Load P-MDP data
    print("Loading P-MDP data...")
    Ex1_PMDP_10AS_WS1_SuccessRate = load_data(pmdp_10as_path)
    Ex1_PMDP_30AS_WS1_SuccessRate = load_data(pmdp_30as_path)
    Ex1_PMDP_50AS_WS1_SuccessRate = load_data(pmdp_50as_path)
    if None in [Ex1_PMDP_10AS_WS1_SuccessRate, Ex1_PMDP_30AS_WS1_SuccessRate, Ex1_PMDP_50AS_WS1_SuccessRate]:
        print("Failed to load P-MDP data, experiment plotting aborted.")
        return

    # Load CSSC-MDP data
    print("Loading CSSC-MDP data...")
    CSSC_10AS_WS1_SuccessRate_200Interval_5wEpisode = load_data(cssc_10as_5w_path)
    CSSC_30AS_WS1_SuccessRate_200Interval_2wEpisode = load_data(cssc_30as_2w_path)
    CSSC_50AS_WS1_SuccessRate_200Interval_2wEpisode = load_data(cssc_50as_2w_path)
    if None in [CSSC_10AS_WS1_SuccessRate_200Interval_5wEpisode, CSSC_30AS_WS1_SuccessRate_200Interval_2wEpisode, CSSC_50AS_WS1_SuccessRate_200Interval_2wEpisode]:
        print("Failed to load CSSC-MDP data, experiment plotting aborted.")
        return

    # -------------------------- 2. Data processing (consistent with original logic) --------------------------
    # Process CSSC 10AS data: Take first 100 elements from 5wEpisode data as 2wEpisode data, then truncate to final range
    CSSC_10AS_WS1_SuccessRate_200Interval_2wEpisode = []
    for item in CSSC_10AS_WS1_SuccessRate_200Interval_5wEpisode:
        CSSC_10AS_WS1_SuccessRate_200Interval_2wEpisode.append(item[:cssc_5w_to_2w_cut])

    CSSC_10AS_WS1_SuccessRate = []
    for item in CSSC_10AS_WS1_SuccessRate_200Interval_2wEpisode:
        CSSC_10AS_WS1_SuccessRate.append(item[cssc_final_cut_start:cssc_final_cut_end])

    # Process CSSC 30AS data: Truncate to final range
    CSSC_30AS_WS1_SuccessRate = []
    for item in CSSC_30AS_WS1_SuccessRate_200Interval_2wEpisode:
        CSSC_30AS_WS1_SuccessRate.append(item[cssc_final_cut_start:cssc_final_cut_end])

    # Process CSSC 50AS data: Truncate to final range
    CSSC_50AS_WS1_SuccessRate = []
    for item in CSSC_50AS_WS1_SuccessRate_200Interval_2wEpisode:
        CSSC_50AS_WS1_SuccessRate.append(item[cssc_final_cut_start:cssc_final_cut_end])

    # -------------------------- 3. Combine data and labels --------------------------
    Experi_2Dlist_SuccessRate = [
        Ex1_PMDP_10AS_WS1_SuccessRate,
        CSSC_10AS_WS1_SuccessRate,
        Ex1_PMDP_30AS_WS1_SuccessRate,
        CSSC_30AS_WS1_SuccessRate,
        Ex1_PMDP_50AS_WS1_SuccessRate,
        CSSC_50AS_WS1_SuccessRate
    ]
    labels = ['P-MDP 10', 'CSSC-MDP 10', 'P-MDP 30', 'CSSC-MDP 30', 'P-MDP 50', 'CSSC-MDP 50']

    # -------------------------- 4. Plot charts in two modes --------------------------
    # 1. Plot in point mode (plot_mode='point')
    plot_Experiment_pictures(
        Experi_2Dlist_SuccessRate, labels,
        title=title, xlabel=xlabel, ylabel=ylabel,
        eval_interval=eval_interval,
        window_size=window_size_point,
        trend_window_size=trend_window_size,
        inset_mode=inset_mode,
        y_compress_main=y_compress_main,
        y_compress_inset=y_compress_inset,
        plot_mode='point',
        color_group_size=color_group_size,
        grid_on=grid_on_point,
        show_error_shadow=show_error_shadow_point,
        x_axis_start=x_axis_start,
        label_mode=label_mode,
        fontsize_increase=fontsize_increase
    )

    # 2. Plot in line mode (plot_mode='line')
    plot_Experiment_pictures(
        Experi_2Dlist_SuccessRate, labels,
        title=title, xlabel=xlabel, ylabel=ylabel,
        eval_interval=eval_interval,
        window_size=window_size_line,
        trend_window_size=trend_window_size,
        inset_mode=inset_mode,
        y_compress_main=y_compress_main,
        y_compress_inset=y_compress_inset,
        plot_mode='line',
        color_group_size=color_group_size,
        grid_on=grid_on_line,
        show_error_shadow=show_error_shadow_line,
        x_axis_start=x_axis_start,
        label_mode=label_mode,
        fontsize_increase=fontsize_increase
    )



def plot_parameter_comparison(
    data_paths,  # List of 5 path dicts for the 5 parameter settings
    labels,      # List of 5 labels corresponding to each parameter setting
    # Plotting parameters
    title='',
    xlabel='Thousands of episodes',
    ylabel='Service Composition Success Rate',
    eval_interval=200,
    window_size=5,
    trend_window_size=5,
    inset_mode='main',
    y_compress_main=(0.0, 0.0),
    y_compress_inset=(0.0, 0.0),
    plot_mode='line',
    color_group_size=1,
    grid_on=False,
    show_error_shadow=True,
    x_axis_start=400,
    line_type='auto',
    zoom_enabled=True,
    zoom_range=[1, 10],
    zoom_y_range=[0.38, 0.76],
    x_range=[0, 10],
    label_mode='manual',
    fontsize_increase=10
):
    """
    Visualize parameter comparison experiments, generate plots, and calculate normalized statistics.
    
    Parameters:
        data_paths (list): List of 5 dictionaries, each containing 'data_dir', 'sub_dir', 'filename' for data files
        labels (list): List of 5 strings for parameter labels (e.g., ['α = 0.00125', ...] or ['ω = 0.1', ...])
        Other parameters: Plotting configurations for plot_Experiment_pictures
    
    Returns:
        list: Normalized statistics including AUC, average success rate, and their normalizations
    """
    # -------------------------- 1. Load experiment data --------------------------
    def load_data(path):
        """Helper function to load data using path parameters"""
        data = load_data_from_file(
            data_dir=path['data_dir'],
            sub_dir=path['sub_dir'],
            filename=path['filename']
        )
        if data is None:
            print(f"Failed to load data from {path}, experiment aborted.")
        return data

    # Load 5 datasets for parameter comparison
    print("Loading parameter comparison data...")
    experiment_data = []
    for i, path in enumerate(data_paths):
        data = load_data(path)
        if data is None:
            print(f"Data loading failed for parameter setting {i+1}, experiment aborted.")
            return None
        experiment_data.append(data)

    # -------------------------- 2. Generate experiment plot --------------------------
    stats_data = plot_Experiment_pictures(
        experiment_data, labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        eval_interval=eval_interval,
        window_size=window_size,
        trend_window_size=trend_window_size,
        inset_mode=inset_mode,
        y_compress_main=y_compress_main,
        y_compress_inset=y_compress_inset,
        plot_mode=plot_mode,
        color_group_size=color_group_size,
        grid_on=grid_on,
        show_error_shadow=show_error_shadow,
        x_axis_start=x_axis_start,
        line_type=line_type,
        zoom_enabled=zoom_enabled,
        zoom_range=zoom_range,
        zoom_y_range=zoom_y_range,
        x_range=x_range,
        label_mode=label_mode,
        fontsize_increase=fontsize_increase
    )

    # -------------------------- 3. Calculate normalized statistics --------------------------
    # Get maximum values for normalization
    try:
        max_auc = max(item['auc'] for item in stats_data)
        max_average_mean = max(item['avg_SR'] for item in stats_data)
        max_average_se_mean = max(item['avg_SE'] for item in stats_data)
    except (KeyError, TypeError) as e:
        print(f"Error processing statistics data: {e}")
        return None

    # Normalize each statistic
    normalized_stats = []
    for item in stats_data:
        normalized_item = item.copy()  # Copy original data to avoid modifying source
        # Handle division by zero
        normalized_item['auc_norm'] = item['auc'] / max_auc if max_auc != 0 else 0
        normalized_item['avg_SR_norm'] = item['avg_SR'] / max_average_mean if max_average_mean != 0 else 0
        normalized_item['avg_SE_norm'] = item['avg_SE'] / max_average_se_mean if max_average_se_mean != 0 else 0
        normalized_stats.append(normalized_item)

    # Print normalized statistics
    #print("\n################## Normalized statistics data: ##################\n")
    #for item in normalized_stats:
        #print(item)

    # Print normalized statistics with selected fields
    print("\n################## Normalized statistics data: ##################\n")
    for item in normalized_stats:
        # Extract the value of ω from the label (assuming format 'ω = X.XX')
        omega_value = item['label'].split('=')[1].strip()
        # Create a new dictionary with only the required fields in order
        filtered_item = {
            'ω': omega_value,
            'auc_norm': item['auc_norm'],
            'avg_SR': item['avg_SR'],
            'avg_SE': item['avg_SE']
        }
        print(filtered_item)
    print("\n\n")

    return normalized_stats
    
def plot_3d_surface(data, 
                    x_values=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 
                    y_values=np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])):
    """
    Create a 3D surface plot from the provided data.
    
    Parameters:
        data (list): 2D list containing the data to be plotted
        x_values (np.ndarray): Array of values for the X-axis (default: [10, 20, ..., 100])
        y_values (np.ndarray): Array of values for the Y-axis (default: [5, 10, ..., 50])
    """
    # Create coordinate grid
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.array(data)

    # Validate data shape (ensure consistency with x and y dimensions)
    if Z.shape != (len(y_values), len(x_values)):
        raise ValueError(f"Data shape ({Z.shape}) does not match x/y dimensions ({len(y_values)}, {len(x_values)})")

    # Create 3D figure
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.5)

    # Setting axis labels
    ax.set_xlabel('Number of Candidate Services', fontsize=15, labelpad=15)
    ax.set_ylabel('Number of Abstract Services', fontsize=15, labelpad=15)
    ax.set_zlabel('Average training time per episode (s)', fontsize=15, labelpad=15)

    # Set axis scales and tick parameters
    ax.set_xticks(x_values)
    ax.set_yticks(y_values)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='x', labelsize=15)  
    ax.tick_params(axis='y', labelsize=15) 
    ax.tick_params(axis='z', labelsize=15)  

    # Add color bar
    cbar = fig.colorbar(surf, shrink=0.6, aspect=18, pad=0.02)
    cbar.ax.tick_params(labelsize=15)

    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    
    
    if os.environ.get('IS_IN_DOCKER') == 'true':
        # for docker env, save picture only
        global plot_counter
        plot_counter += 1
        filename = f"plot_result_{plot_counter}.png" # e.g.,: plot_result_1.png, plot_result_2.png

        output_path = os.path.join("/app/src/training_records", filename)

        plt.savefig(output_path)
        print(f"Running in Docker. Plot saved to {output_path}")
    else:
        # 如果是在本地环境 (或其他没有设置该变量的环境)，则显示图片
        print("Running in local environment. Displaying plot.")

        # Adjust layout and display
        plt.tight_layout()
        plt.show()



def run_experiment_and_plot(
    qws_data_path_params,
    travel_data_path_params,
    plot_title='',
    xlabel='Thousands of episodes',
    ylabel='Rollout Average Rewards',
    eval_interval=200,
    window_size=1,
    trend_window_size=5,
    inset_mode='main',
    y_compress_main=(0.0, 0.0),
    y_compress_inset=(0.0, 0.0),
    plot_mode='line',
    color_group_size=1,
    grid_on=True,
    show_error_shadow=True,
    x_axis_start=0,
    fontsize_increase=10,
    legend_position='lower right'
):
    """
    Load QWS and TravelAgency data, process it, and generate experiment plots.
    
    Parameters:
        qws_data_path_params (dict): Parameters for loading QWS data (data_dir, sub_dir, filename)
        travel_data_path_params (dict): Parameters for loading TravelAgency data (data_dir, sub_dir, filename)
        Other parameters: Plotting configuration (same as plot_Experiment_pictures)
    
    Returns:
        None
    """
    def update_2d_list(lst):
        """Truncate each sublist to first 50 elements"""
        for i in range(len(lst)):
            lst[i] = lst[i][:50]
        return lst

    # Load and process QWS data
    print("Loading QWS data...")
    avg_QWS_data = load_data_from_file(
        data_dir=qws_data_path_params['data_dir'],
        sub_dir=qws_data_path_params['sub_dir'],
        filename=qws_data_path_params['filename']
    )
    if avg_QWS_data is None:
        print("Failed to load QWS data, exiting experiment")
        return
    QWS_avg_data = update_2d_list(avg_QWS_data)

    # Load and process TravelAgency data
    print("Loading TravelAgency data...")
    avg_travel_data = load_data_from_file(
        data_dir=travel_data_path_params['data_dir'],
        sub_dir=travel_data_path_params['sub_dir'],
        filename=travel_data_path_params['filename']
    )
    if avg_travel_data is None:
        print("Failed to load TravelAgency data, exiting experiment")
        return
    Travel_avg_data = update_2d_list(avg_travel_data)

    # Prepare experiment data and generate plot
    experi_2_data = [Travel_avg_data, [[]], QWS_avg_data]
    plot_Experiment_pictures(
        experi_2_data,
        ['P-MDP for TravelAgency', '', 'P-MDP for QWS'],
        title=plot_title,
        xlabel=xlabel,
        ylabel=ylabel,
        eval_interval=eval_interval,
        window_size=window_size,
        trend_window_size=trend_window_size,
        inset_mode=inset_mode,
        y_compress_main=y_compress_main,
        y_compress_inset=y_compress_inset,
        plot_mode=plot_mode,
        color_group_size=color_group_size,
        grid_on=grid_on,
        show_error_shadow=show_error_shadow,
        x_axis_start=x_axis_start,
        fontsize_increase=fontsize_increase,
        legend_position=legend_position
    )

#### For plotting cumulative rewards of QWS and TravelAgency experiments, Fig.6a in submmited paper###
def plot_cumulative_rewards(
    qws_rewards_params,
    travel_rewards_params,
    xlabel='Thousands of episodes',
    ylabel='Cumulative Rewards',
    eval_interval=1,
    window_size=5,
    trend_window_size=5,
    inset_mode='main',
    y_compress_main=(0.0, 0.0),
    y_compress_inset=(0.0, 0.0),
    plot_mode='line',
    color_group_size=1,
    grid_on=True,
    show_error_shadow=True,
    x_axis_start=0,
    fontsize_increase=8,
    legend_position='lower right',
    figsize=(10, 8)
):
    """
    Plot cumulative rewards for QWS and TravelAgency experiments in subplots.
    
    Parameters:
        qws_rewards_params (dict): Path parameters for QWS accumulated rewards data
            (keys: data_dir, sub_dir, filename)
        travel_rewards_params (dict): Path parameters for TravelAgency accumulated rewards data
            (keys: data_dir, sub_dir, filename)
        xlabel (str): X-axis label for both subplots
        ylabel (str): Y-axis label for both subplots
        figsize (tuple): Figure size (width, height)
        Other parameters: Shared plotting configurations for plot_Experiment_pictures
    
    Returns:
        None
    """
    # 1. Load QWS data
    print("Loading QWS accumulated rewards data...")
    QWS_accumulated_total_rewards = load_data_from_file(
        data_dir=qws_rewards_params['data_dir'],
        sub_dir=qws_rewards_params['sub_dir'],
        filename=qws_rewards_params['filename']
    )
    if QWS_accumulated_total_rewards is None:
        print("Failed to load QWS data, exiting plot function")
        return
    
    # 2. Load TravelAgency data
    print("Loading TravelAgency accumulated rewards data...")
    travel_accumulated_total_rewards = load_data_from_file(
        data_dir=travel_rewards_params['data_dir'],
        sub_dir=travel_rewards_params['sub_dir'],
        filename=travel_rewards_params['filename']
    )
    if travel_accumulated_total_rewards is None:
        print("Failed to load TravelAgency data, exiting plot function")
        return
    
    # 3. Prepare data and labels
    QWS_data = [[[]], [[]], QWS_accumulated_total_rewards]
    QWS_labels = ['', '', 'P-MDP for QWS']
    
    Travel_data = [travel_accumulated_total_rewards]
    Travel_labels = ['P-MDP for TravelAgency']
    
    # 4. Create subplots and plot
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot TravelAgency data in first subplot
    plot_Experiment_pictures(
        Travel_data, Travel_labels,
        title='', xlabel=xlabel, ylabel=ylabel,
        eval_interval=eval_interval, window_size=window_size,
        trend_window_size=trend_window_size, inset_mode=inset_mode,
        y_compress_main=y_compress_main, y_compress_inset=y_compress_inset,
        plot_mode=plot_mode, color_group_size=color_group_size, grid_on=grid_on,
        show_error_shadow=show_error_shadow, x_axis_start=x_axis_start,
        ax=axes[0], fontsize_increase=fontsize_increase,
        legend_position=legend_position
    )
    
    # Plot QWS data in second subplot
    plot_Experiment_pictures(
        QWS_data, QWS_labels,
        title='', xlabel=xlabel, ylabel=ylabel,
        eval_interval=eval_interval, window_size=window_size,
        trend_window_size=trend_window_size, inset_mode=inset_mode,
        y_compress_main=y_compress_main, y_compress_inset=y_compress_inset,
        plot_mode=plot_mode, color_group_size=color_group_size, grid_on=grid_on,
        show_error_shadow=show_error_shadow, x_axis_start=x_axis_start,
        ax=axes[1], fontsize_increase=fontsize_increase,
        legend_position=legend_position
    )
    
    if os.environ.get('IS_IN_DOCKER') == 'true':
        # for docker env, save picture only
        global plot_counter
        plot_counter += 1
        filename = f"plot_result_{plot_counter}.png" # e.g.,: plot_result_1.png, plot_result_2.png

        output_path = os.path.join("/app/src/training_records", filename)

        plt.savefig(output_path)
        print(f"Running in Docker. Plot saved to {output_path}")
    else:
        # 如果是在本地环境 (或其他没有设置该变量的环境)，则显示图片
        print("Running in local environment. Displaying plot.")
    
        # Adjust layout and show
        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":

    """
    Main execution script for visualizing experimental results using core utility functions.
    
    Core procedure with Key Functions:
    1. Data Loading with load_data_from_file():
       - Purpose: Loads 2D list data (e.g., [[v11, v12...], [v21, v22...]]) from text files
       - Usage: 
         data = load_data_from_file(
             data_dir="root_data_folder",
             sub_dir="subfolder",
             filename="data_file.txt"
         )
       - Notes: Ensure file contains valid nested lists with numerical values (success rates, rewards, etc.)
    
    2. Visualization with plot_Experiment_pictures():
       - Purpose: Plots experimental data (lines/points) with metrics (AUC, averages) calculation
       - Usage:
         stats = plot_Experiment_pictures(
             instance_evaluation_rewards_2Dlist=data_list,  # List of 2D data (one per curve)
             labels=["Label1", "Label2"],                  # Curve labels
             plot_mode="line",                             # "line" or "point"
             x_range=(start, end)                          # Calculate metrics in this x-interval
         )
       - Key Features: Error shadows, zoomed regions, trend subplots, and customizable styling


    This script executes multiple visualization tasks using the above functions:
    1. Benchmark comparison between P-MDP and CSSC-MDP on WSDREAM dataset (Fig.5a-b)
    2. Parameter sensitivity analysis (learning rate α and PER ω) (Fig.6b-c)
    3. 3D surface plot of training time vs. candidate/abstract services (Fig.6d)
    4. Cumulative rewards for QWS and TravelAgency datasets (Fig.7a)
    5. Rollout average rewards comparison for QWS and TravelAgency (Fig.7b)


    Usage Instructions:
    - Ensure all data files exist in the specified paths (adjust 'data_dir', 'sub_dir', 'filename' in parameters)
    - Run the script directly; plots will be generated sequentially
    - Modify parameters (e.g., zoom ranges, labels) in each function call to customize visualizations
    """

    ######### Start of Plot the results of the P-MDP and CSSC-MDP benchmark experiments from the WSDREAM QoS dataset 2. #########

    #### Fig.5a-b in the submitted paper ####
    # Define path parameters for all data files (adjust according to actual project structure)
    path_params = {
        # P-MDP data paths (10AS, 30AS, 50AS)
        'pmdp_10as': {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_10AS_success_rate.txt'
        },
        'pmdp_30as': {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_30AS_success_rate.txt'
        },
        'pmdp_50as': {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_50AS_success_rate.txt'
        },
        # CSSC-MDP data paths (10AS 5wEpisode, 30AS 2wEpisode, 50AS 2wEpisode)
        'cssc_10as_5w': {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'CSSC_10AS_Success_rate_5w_episode.txt'
        },
        'cssc_30as_2w': {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'CSSC_30AS_Success_rate_2w_episode.txt'
        },
        'cssc_50as_2w': {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'CSSC_50AS_Success_rate_2w_episode.txt'
        }
    }

    # Run comparison experiment and plot results
    run_comparison_experiment(
        pmdp_10as_path=path_params['pmdp_10as'],
        pmdp_30as_path=path_params['pmdp_30as'],
        pmdp_50as_path=path_params['pmdp_50as'],
        cssc_10as_5w_path=path_params['cssc_10as_5w'],
        cssc_30as_2w_path=path_params['cssc_30as_2w'],
        cssc_50as_2w_path=path_params['cssc_50as_2w']
        # Other data processing or plotting parameters can be adjusted as needed
    )

    ########### End of Plot the results of the P-MDP and CSSC-MDP benchmark experiments from the WSDREAM QoS dataset 2. #########

    ################# Start of Plot the results of the parameter comparison experiments ################
    # -------------------------- 1. Learning Rate Comparison (α) --------------------------
    print("Running learning rate comparison experiment...")
    # Path parameters for learning rate experiment data (Ex2_PMDP_10AS_SuccessRate_1 to 5)
    lr_data_paths = [
        {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_convergence_comparison_over_param_LR_1.txt'  # α = 0.00125
        },
        {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_convergence_comparison_over_param_LR_2.txt'  # α = 0.0025
        },
        {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_convergence_comparison_over_param_LR_3.txt'  # α = 0.00375
        },
        {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_convergence_comparison_over_param_LR_4.txt'  # α = 0.005
        },
        {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_convergence_comparison_over_param_LR_5.txt'  # α = 0.00625
        }
    ]
    lr_labels = ['α = 0.00125', 'α = 0.0025', 'α = 0.00375', 'α = 0.005', 'α = 0.00625']
    
    # Run learning rate experiment
    lr_normalized_stats = plot_parameter_comparison(
        data_paths=lr_data_paths,
        labels=lr_labels,
        zoom_range=[1, 10],
        zoom_y_range=[0.38, 0.76],
        x_range=[0, 10]
    )


    # -------------------------- 2. PER ω Comparison --------------------------
    print("\nRunning PER ω comparison experiment...")
    # Path parameters for PER ω experiment data (Ex3_PMDP_10AS_SuccessRate_1 to 5)
    per_data_paths = [
        {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_convergence_comparison_over_PER_ω_1.txt'  # ω = 0.1
        },
        {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_convergence_comparison_over_PER_ω_2.txt'  # ω = 0.3
        },
        {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_convergence_comparison_over_PER_ω_3.txt'  # ω = 0.5
        },
        {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_convergence_comparison_over_PER_ω_4.txt'  # ω = 0.7
        },
        {
            'data_dir': 'Experiment_results',
            'sub_dir': 'WSDREAM',
            'filename': 'PMDP_convergence_comparison_over_PER_ω_5.txt'  # ω = 0.9
        }
    ]
    per_labels = ['ω = 0.1', 'ω = 0.3', 'ω = 0.5', 'ω = 0.7', 'ω = 0.9']
    
    # Run PER ω experiment
    per_normalized_stats = plot_parameter_comparison(
        data_paths=per_data_paths,
        labels=per_labels,
        zoom_range=[10, 20],
        zoom_y_range=[0.64, 0.78],
        x_range=[10, 20]
    )

    ############### End of Plot the results of the parameter comparison experiments ################

    
    ################ Start of the experiment for the tranining time (each-episode) in different candidate services and abstract services ################

    data = load_data_from_file(data_dir="Experiment_results", sub_dir="WSDREAM", filename="Training_time_per_episode.txt")
    if data is not None:
        # Convert the data to a numpy array for plotting
        data_array = np.array(data)
        plot_3d_surface(data_array)

    ################ End of the experiment for the tranining time (each-episode) in different candidate services and abstract services ################




    ################# Start of Plot the results of the experiment for QWS and TravelAgency ################

    ### Fig.7a in the submitted paper ###
    # This part is for plotting cumulative rewards of QWS and TravelAgency experiments
    # Define path parameters for data files
    qws_params = {
        'data_dir': 'Experiment_results',
        'sub_dir': 'QWS',
        'filename': 'QWS_accumulated_total_rewards.txt'
    }
    
    travel_params = {
        'data_dir': 'Experiment_results',
        'sub_dir': 'TravelAgency',
        'filename': 'TravelAgency_accumulated_total_rewards.txt'
    }
    
    # Generate the plot
    plot_cumulative_rewards(
        qws_rewards_params=qws_params,
        travel_rewards_params=travel_params,
        xlabel='Thousands of episodes',
        ylabel='Cumulative Rewards',
        figsize=(10, 8)
        # Adjust other plotting parameters as needed
    )


    ### Fig.6b in the submitted paper ###
    # This part is for plotting rollout average rewards of QWS and TravelAgency experiments
    # Define path parameters for both datasets
    qws_params = {
        'data_dir': 'Experiment_results',
        'sub_dir': 'QWS',
        'filename': 'QWS_Rollout_average_rewards_of_P-MDP.txt'
    }
    
    travel_params = {
        'data_dir': 'Experiment_results',
        'sub_dir': 'TravelAgency',
        'filename': 'TravelAgency_Rollout_average_rewards_of_P-MDP.txt'
    }

    # Run experiment and generate plot
    run_experiment_and_plot(
        qws_data_path_params=qws_params,
        travel_data_path_params=travel_params
        # Other plotting parameters can be adjusted here
    )
    
    ################# End of the experiment for QWS and TravelAgency ################

