// Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
#include "AlignedReID.h"
#include <unistd.h>
#include <set>
#include <map>
#include <regex>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <vector>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

void softmax(const std::vector<float>& src, std::vector<float>& dst, int length);
void save_result(std::string file_name, int inferred_pid, std::string res_path);
#define RESULT_FILE "./result.json"
#define GALLERY_PATH "../data/test/"
void save_result(std::string file_name, int inferred_pid);

APP_ERROR AlignedReID::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR AlignedReID::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR AlignedReID::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR AlignedReID::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    static constexpr uint32_t resizeHeight = 256;
    static constexpr uint32_t resizeWidth = 128;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR AlignedReID::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize =  imageMat.cols *  imageMat.rows * YUV444_RGB_WIDTH_NU;
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(imageMat.data, dataSize, MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {imageMat.rows * YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR AlignedReID::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                       std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t) modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    dynamicInfo.batchSize = 1;

    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR AlignedReID::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                         std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
                                         const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                                         const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
    APP_ERROR ret = post_->Process(inputs, objectInfos, resizedImageInfos, configParamMap);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR AlignedReID::Process(const std::string &imgPath, std::vector<float>& features) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);

    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    ResizeImage(imageMat, imageMat);

    TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, tensorBase);

    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(tensorBase);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    /* Get result from stream */
    MxBase::TensorBase& tensor0 = outputs.at(1);
    auto outputShape0 = tensor0.GetShape();
    uint32_t  classNum0 = outputShape0[1];
    void* data0 = tensor0.GetBuffer();
    for (uint32_t i = 0; i < classNum0; i++) {
        float value0 = *(reinterpret_cast<float*>(data0) + i);
        features.push_back(value0);
    }

    return APP_ERR_OK;
}


void softmax(const std::vector<float>& src, std::vector<float>& dst, int length) {
    float alpha = *std::max_element(std::begin(src), std::end(src));
    float denominator{ 0 };
    for (int i = 0; i < length; ++i) {
        dst[i] = std::exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
        dst[i] /= denominator;
}

void save_result(std::vector<int> query_pids, std::vector<int> inferred_pids) {
    std::ofstream fout(RESULT_FILE, ios::app | ios::out);
    for (unsigned i = 0; i < query_pids.size(); i++) {
        fout << "query pid: " << setw(7) << query_pids[i] << "  inferred pid: " << setw(7) << inferred_pids[i] << "\n";
    }
    fout.close();
}
