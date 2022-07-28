import torchvision
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def make_plot(ctx, tgt, pred, epoch, cmap='gray'):
    num_ctx_frames= ctx.shape[1]
    num_tgt_frames = tgt.shape[1]

    def show_frames(frames, ax, row_label=None):
        for i, frame in enumerate(frames):
            if cmap is not None:
                ax[i].imshow(frame, cmap)
            else:
                ax[i].imshow(frame)
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        if row_label is not None:
            ax[0].set_ylabel(row_label)

    ctx_frames = ctx.squeeze().cpu().numpy()
    tgt_frames = tgt.squeeze().cpu().numpy()
    pred_frames = pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(3, max(num_ctx_frames, num_tgt_frames),
                       figsize = (9, 5))
    fig.suptitle(f"EPOCH {epoch}", y=0.93)
    show_frames(ctx_frames, ax[0], "Context")
    show_frames(tgt_frames, ax[1], "Target")
    show_frames(pred_frames, ax[2], "Prediction")

    return fig

def fig2image(fig):
    # fig.canvas.draw()
    # buf = fig.canvas.tostring_rgb()
    # ncols, nrows = fig.canvas.get_width_height()
    # shp = (nrows, ncols, 3)
    # arr = np.frombuffer(buf, dtype=np.uint8).reshape(shp)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    img = Image.open(buf)
    img = torchvision.transforms.ToTensor()(img)
    return img

def make_plot_image(ctx, tgt, pred, epoch, cmap='gray'):
    return fig2image(make_plot(ctx, tgt, pred, epoch, cmap='gray'))