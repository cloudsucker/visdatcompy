from utils import scan_directory
import metrics

def check_image(image: str = None, images: list = None, metric: str = None, fast_checking=False):
    metric_list = {
        "pix2pix":metrics.pix2pix,
        "mse":metrics.mse,
        "ssim":metrics.ssim,
        "psnr":metrics.psnr,
        "mae":metrics.mae,
    }
    current_metric = metric_list[metric]
    #current_metric(image, images)
    
    return


if __name__ == "__main__":
    print(check_image(metric='pix2pix'))
