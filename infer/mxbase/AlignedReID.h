// Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
#ifndef A_D
#define A_D

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "ObjectPostProcessors/SsdMobilenetFpnMindsporePost.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    float iou_thresh;
    float score_thresh;
    bool checkTensor;
    std::string modelPath;
};

class AlignedReID {
 public:
     APP_ERROR Init(const InitParam &initParam);
     APP_ERROR DeInit();
     APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat);
     APP_ERROR ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
     APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);
     APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
     APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
         std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
         const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
         const std::map<std::string, std::shared_ptr<void>> &configParamMap);
     APP_ERROR Process(const std::string &imgPath, std::vector<float> &features);
 private:
     std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
     std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
     std::shared_ptr<MxBase::SsdMobilenetFpnMindsporePost> post_;
     MxBase::ModelDesc modelDesc_;
     uint32_t deviceId_ = 0;
};
#endif
