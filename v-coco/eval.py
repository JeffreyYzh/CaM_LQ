from vsrl_eval import VCOCOeval
import utils
import pdb

if __name__ == "__main__":
    vsrl_annot_file = "data/vcoco/vcoco_test.json"
    coco_file = "data/instances_vcoco_all_2014.json"
    split_file = "data/splits/vcoco_test.ids"

    # Change this line to match the path of your cached file
    det_file = "cache.pkl"
    pdb.set_trace()
    # print(f"Loading cached results from {det_file}.")
    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    vcocoeval._do_eval(det_file, ovr_thresh=0.5)