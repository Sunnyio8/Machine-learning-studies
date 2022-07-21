# NMS（非极大值抑制）

Created: July 21, 2022 3:13 PM
Reviewed: No

## NMS的原理和示例

NMS全程为非极大值抑制，当目标检测网络产生输出后，单个目标可能输出多个检测框，NMS方法的作用为去除同个目标中**IOU重叠度较高**的、**置信度score较低**的那些检测框

下图为NMS的使用效果：

[https://camo.githubusercontent.com/60b1be397ce7bf654c5020e4255dab12562a9b12cd97ba1704d5d3d7effb078e/68747470733a2f2f692e6c6f6c692e6e65742f323032302f30352f31352f6774587251437677793235624f684d2e6a7067](https://camo.githubusercontent.com/60b1be397ce7bf654c5020e4255dab12562a9b12cd97ba1704d5d3d7effb078e/68747470733a2f2f692e6c6f6c692e6e65742f323032302f30352f31352f6774587251437677793235624f684d2e6a7067)

NMS中的**IOU**（交并比）也是一个重要的函数

实现NMS的流程为：

| 目标框 | 置信度(score) |
| --- | --- |
| A | 0.9 |
| B | 0.8 |
| C | 0.7 |
| D | 0.6 |
| E | 0.5 |
| F | 0.4 |

(1)首先按照六个目标框的置信度进行排序

(2)取出最大置信度的框A并保存

(3)分别对剩余的框计算与A的IOU，如果IOU大于我们设置的阈值（一般为0.5），则将该目标框丢弃

(4)重复以上流程，直至排序序列为空

## NMS的缺点：

- **需要手动设置阈值，阈值的设置会直接影响重叠目标的检测**
- **IOU超过阈值的直接设置score为0，做法太hard**
- **通过IoU来评估，IoU的做法对目标框尺度和距离的影响不同**

## 改进版本soft-nms

soft-NMS就是用一个稍微小一点的分数替代原有的分数，而非直接粗暴的置0

将当前检测框得分乘以一个权重函数，该函数会衰减与最高得分检测框M有重叠的相邻检测框的分数，越是与M框高度重叠的检测框，其得分衰减越严重，为此选择高斯函数为权重函数，从而修改其删除检测框的规则。高斯权重函数如下所示（δ通常取0.3）。

![https://img-blog.csdnimg.cn/20210224223337682.png#pic_center](https://img-blog.csdnimg.cn/20210224223337682.png#pic_center)

## NMS的代码实现

```python
def single_nms(bboxes, scores, thresh=0.5):
"""
NMS

Parameters:
  bboxes - shape:[N,4]
  scores - shape:[N,1]
  thresh - 阈值

Returns:
    

"""
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    
    #计算检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    #按照升序对score进行排序
    order = scores.argsort()
    #将升序转换为降序
    order = order[::-1]
    
    #结果
    result = []
    
    while order.size() > 0:
        if order.size() == 1:
            i = order[0]
            keep.append(i)
            break
        else:
            i = order[0]
            keep.append(i)
        
        #计算相交区域的左上坐标及右下坐标
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # 计算相交的面积，不重叠时为0
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        #计算IOU
        iou = inter /  (arears[i] + areas[order[1:]] - inter)
        
        #保留IOU小于阈值的bboxes
        inds = np.where(iou <= thresh)[0]
        if inds.size() == 0:
            break
        order = order[inds + 1]# 因为我们上面求iou的时候得到的结果索引与order相比偏移了一位，因此这里要补回来
    return keep
```

## pytorch版本实现

```python
def NMS(bboxes, scores, thresh=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2-x1+1)*(y2-y1+1)

    #torch.sort() descending=True:降序
    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.size() == 1:
            break
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w*h

        iou = inter/(areas[i] + areas[order[1:]] - inter)
        ids = (iou < thresh).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
```