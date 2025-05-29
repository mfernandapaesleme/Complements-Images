import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class ImageViewer:
    def __init__(self, results_data):
        self.results_data = results_data
        self.current_index = 0
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.subplots_adjust(bottom=0.15)
        
        # Create navigation buttons
        ax_prev = plt.axes([0.3, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.6, 0.05, 0.1, 0.04])
        ax_info = plt.axes([0.45, 0.05, 0.1, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_info = Button(ax_info, 'Summary')
        
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_info.on_clicked(self.show_summary)
        
        # Display first image
        self.update_display()
        
    def update_display(self):
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
            ax.axis('off')
        
        if not self.results_data:
            return
            
        data = self.results_data[self.current_index]
        
        # Update title
        self.fig.suptitle(f"Image {self.current_index + 1}/{len(self.results_data)}: {data['name']}\n"
                         f"Accuracy: {data['accuracy']:.4f}, Recall: {data['recall']:.4f}", 
                         fontsize=14)
        
        # Display images
        self.axes[0, 0].imshow(data['img'], cmap='gray')
        self.axes[0, 0].set_title('Image Originale')
        
        self.axes[0, 1].imshow(data['img_out'])
        self.axes[0, 1].set_title('Segmentation')
        
        self.axes[0, 2].imshow(data['img_out_skel'])
        self.axes[0, 2].set_title('Segmentation squelette')
        
        self.axes[1, 0].imshow(data['img_GT'])
        self.axes[1, 0].set_title('Verite Terrain')
        
        self.axes[1, 1].imshow(data['GT_skel'])
        self.axes[1, 1].set_title('Verite Terrain Squelette')
        
        # Create overlay comparison in the last subplot
        overlay = np.zeros((*data['img_GT'].shape, 3))
        overlay[:, :, 0] = data['img_out_skel']  # Red for segmentation
        overlay[:, :, 1] = data['GT_skel']       # Green for ground truth
        self.axes[1, 2].imshow(overlay)
        self.axes[1, 2].set_title('Overlay (Red: Seg, Green: GT)')
        
        # Remove axes for all subplots
        for ax in self.axes.flat:
            ax.axis('off')
        
        plt.draw()
    
    def prev_image(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def next_image(self, event):
        if self.current_index < len(self.results_data) - 1:
            self.current_index += 1
            self.update_display()
    
    def show_summary(self, event):
        # Create summary statistics window
        summary_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        accuracies = [data['accuracy'] for data in self.results_data]
        recalls = [data['recall'] for data in self.results_data]
        names = [data['name'].replace('star', '').replace('.jpg', '') for data in self.results_data]
        
        ax1.bar(range(len(accuracies)), accuracies)
        ax1.set_xlabel('Image Index')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy per Image')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45)
        
        ax2.bar(range(len(recalls)), recalls)
        ax2.set_xlabel('Image Index')
        ax2.set_ylabel('Recall')
        ax2.set_title('Recall per Image')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45)
        
        summary_fig.suptitle(f'Summary Statistics\n'
                           f'Avg Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n'
                           f'Avg Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}')
        
        plt.tight_layout()
        plt.show()
    
    def show(self):
        """Display the viewer window"""
        plt.show()