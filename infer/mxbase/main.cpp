// Copyright (C) 2021.Huawei Technologies Co., Ltd. All rights reserved.
#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "AlignedReID.h"
#include <set>
#include <map>
#include <regex>
#include <cstring>
#include <cstdio>
#include <string>
#include <cmath>
#include "AlignedReID.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NUM = 751;
}
std::map<int, int> mapping();
#define QUERY_PATH "/home/data/sdz_mindx/aligned/infer/data/infer/query/"
#define GALLERY_PATH_ "/home/data/sdz_mindx/aligned/infer/data/test/"
#define RESULT_FILE_ "/home/data/sdz_mindx/aligned/infer/mxbase/result.json"
#define MODEL_PATH "/home/data/sdz_mindx/aligned/infer/data/model/AlignedReID.om"

int infer(const std::string& argv1, const std::string& argv2);
void mul(std::vector<std::vector<float>& > distmat);
std::vector<std::vector<float> > calculate_distmat();

std::vector<int> query_pids;
std::vector<int> gallery_pids;
unsigned query_num = 0;
unsigned gallery_num = 0;
const unsigned FEATURE_SIZE = 2048;
std::vector<std::vector<float> > qf(3500);
std::vector<std::vector<float> > gf(20000);
void normalize();

int main(int argc, char *argv[]) {
    std::vector<std::vector<float> > distmat;


    int ret = infer(MODEL_PATH, QUERY_PATH);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    LogInfo << "==================================";
    LogInfo << "Inference success, qf.size = " << query_num << "  gf.size = " << gallery_num;
    LogInfo << "==================================";

    distmat = calculate_distmat();

    LogInfo << "==================================";
    LogInfo << " All data has been loaded";
    LogInfo << " query_size = " << query_pids.size();
    LogInfo << " gallery_size = " << gallery_pids.size();
    LogInfo << "==================================";
    std::ofstream fout(RESULT_FILE_);
    for (unsigned i = 0; i < query_num; i++) {
        float min_element = 1000;
        unsigned min_index = -1;
        for (unsigned j = 0; j < gallery_num; j++) {
            min_element = std::min(min_element, distmat[i][j]);
            if (distmat[i][j] == min_element)
                min_index = j;
        }
        int query_pid = query_pids[i];
        int inferred_pid = gallery_pids[min_index];

        std::cout << "  query_pid: " << query_pid;
        std::cout.width(8 - static_cast<int>(log10(query_pid) + 1));
        std::cout << " -> " << " inferred_pid: " << inferred_pid;
        std::cout.width(8 - static_cast<int>(log10(inferred_pid) + 1));
        std::cout << "   img_index: " << min_index << "\n";
        fout << "query pid: " << query_pid << "  inferred pid: " << inferred_pid << "\n";
    }
    fout.close();
    LogInfo << "==================================";
}

void normalize() {
    LogInfo << " qf normalization......";
    for (unsigned i = 0; i < m; i++) {
        float sum = 0;
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            sum += pow(qf[i][j], 2);
        float sqt = sqrt(sum);
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            qf[i][j] /= (sqt + 1);
    }
    LogInfo << " gf normalization......";
    // gf normalization
    for (unsigned i = 0; i < n; i++) {
        float sum = 0;
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            sum += pow(gf[i][j], 2);
        float sqt = sqrt(sum);
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            gf[i][j] /= (sqt + 1);
    }
}

void mul(std::vector<std::vector<float>& > distmat) {
    for (unsigned i = 0; i < m; i++) {
        if (i % 336 == 0)
            std::cout << " doing distmat multiplication ----------------->\n";
        for (unsigned j = 0; j < n; j++) {
            float num = 0;
            for (unsigned k = 0; k < FEATURE_SIZE; k++)
                num += qf[i][k] * gf[j][k];
            distmat[i][j] += (-2) * num;
        }
    }
    std::cout << " doing operations have been done ---------------\n";

    LogInfo << "==================================";
}

std::vector<std::vector<float> > calculate_distmat() {
    unsigned m = query_num;
    unsigned n = gallery_num;
    LogInfo << "==================================";
    LogInfo << "qf.size = " << m;
    LogInfo << "gf.size = " << n;
    normalization();

    std::vector<float> a(m);
    for (unsigned i = 0; i < m; i++) {
        a[i] = 0;
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            a[i] += pow(qf[i][j], 2);
    }
    std::vector<float> b(n);
    for (unsigned i = 0; i < n; i++) {
        b[i] = 0;
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            b[i] += pow(gf[i][j], 2);
    }

    std::vector<std::vector<float> > t1(m);
    for (unsigned i = 0; i < m; i++) {
        for (unsigned j = 0; j < n; j++)
            t1[i].push_back(a[i]);
    }

    std::vector<std::vector<float> > t2(m);
    for (unsigned i = 0; i < m; i++) {
        for (unsigned j = 0; j < n; j++)
            t2[i].push_back(b[j]);
    }

    std::vector<std::vector<float> > distmat(m);
    LogInfo << " Doing distmat addition......";
    for (unsigned i = 0; i < m; i++) {
        for (unsigned j = 0; j < n; j++) {
            distmat[i].push_back(0);
            distmat[i][j] = t1[i][j] + t2[i][j];
        }
    }

    LogInfo << " Doing distmat multiplication......";

    mul(distmat);
    return distmat;
}

int infer(const std::string& argv1, const std::string& argv2) {
    std::ofstream fout(RESULT_FILE_);
    fout.close();

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;

    initParam.iou_thresh = 0.6;
    initParam.score_thresh = 0.6;
    initParam.checkTensor = true;

    initParam.modelPath = argv1;
    auto AlignedReID1 = std::make_shared<AlignedReID>();
    APP_ERROR ret = AlignedReID1->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "AlignedReID init failed, ret=" << ret << ".";
        return ret;
    }
    std::regex pattern("([-\\d]+)");

    /********************************************************/
    /*             Handle the query images                  */
    /********************************************************/
    std::string PATH = argv2;
    struct dirent *ptr;
    DIR *dir = opendir(PATH.c_str());
    std::vector<std::string> files;
    while ((ptr = readdir(dir)) != NULL)
        files.push_back(ptr->d_name);
    std::smatch result;
    LogInfo << "==================================";
    LogInfo << "Getting query features......";
    LogInfo << "Number of query images = " << files.size();
    for (unsigned int i = 0; i < files.size(); ++i) {
        /**********************/
        /* Get the query pids */
        int pos = files[i].find_last_of('.');
        std::string suffix(files[i].substr(pos + 1));
        if (suffix != "jpg")
            continue;
        string::const_iterator iterStart = files[i].begin();
        string::const_iterator iterEnd = files[i].end();
        if (regex_search(iterStart, iterEnd, result, pattern)) {
            int pid = atoi(string(result[0]).c_str());
            query_pids.push_back(pid);
        }
        /**********************/
        std::string file_name = PATH + files[i];
        // std::vector<float> temp;
        ret = AlignedReID1->Process(file_name, qf[query_num]);

        if (ret != APP_ERR_OK) {
            LogError << "AlignedReID process failed, ret=" << ret << ".";
            AlignedReID1->DeInit();
            return ret;
        }
        query_num++;
    }
    LogInfo << " 100% have done ";
    LogInfo << "Extracted features from query set, obtained " << query_num << "-by-" << qf[0].size() << "matrix";
    LogInfo << "==================================";

    /********************************************************/
    /*            Handle the gallery images                 */
    /********************************************************/
    PATH = GALLERY_PATH_;
    dir = opendir(PATH.c_str());
    files.clear();
    while ((ptr = readdir(dir)) != NULL)
        files.push_back(ptr->d_name);

    std::smatch result_;
    LogInfo << "==================================";
    LogInfo << "Getting gallery features......";
    LogInfo << "Number of gallery images = " << files.size();
    for (unsigned int i = 0; i < files.size(); ++i) {
        // Get the gallery pids
        int pos = files[i].find_last_of('.');
        std::string suffix(files[i].substr(pos + 1));
        if (suffix != "jpg")
            continue;
        string::const_iterator iterStart = files[i].begin();
        string::const_iterator iterEnd = files[i].end();
        if (regex_search(iterStart, iterEnd, result_, pattern)) {
            int pid = atoi(string(result_[0]).c_str());
            if (pid < 0 || pid > 1501)
                continue;
            gallery_pids.push_back(pid);
        }
        std::string file_name = PATH + files[i];
        ret = AlignedReID1->Process(file_name, gf[gallery_num]);
        if (ret != APP_ERR_OK) {
            LogError << "AlignedReID process failed, ret=" << ret << ".";
            AlignedReID1->DeInit();
            return ret;
        }
        gallery_num++;
    }
    LogInfo << " 100% have done ";
    LogInfo << "Extracted features from gallery set, obtained " << gallery_num << "-by-" << gf[0].size() << "matrix";
    LogInfo << "==================================";

    AlignedReID1->DeInit();
    return APP_ERR_OK;
}
