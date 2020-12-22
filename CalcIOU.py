
# box : [top, left, bottom, right]
def calculate_iou(gt_box, pred_box):
    Y_min, X_min, Y_max, X_max = [int(temp) for temp in gt_box]
    y_min, x_min, y_max, x_max = [int(temp) for temp in pred_box]

    intersection_area = (min(x_max, X_max)-max(x_min, X_min))*(min(y_max, Y_max)-max(y_min, Y_min))
    gt_area = (X_max-X_min)*(Y_max-Y_min)
    pred_area = (x_max-x_min)*(y_max-y_min)

    # print(intersection_area)
    # print(gt_area)
    # print(pred_area)

    iou = intersection_area/float(gt_area+pred_area-intersection_area)

    print(intersection_area)
    print(gt_area)
    print(pred_area)
    return iou
