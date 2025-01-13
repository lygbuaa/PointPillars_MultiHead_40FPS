// headers in STL
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
// headers in 3rd-part
#include "../pointpillars/pointpillars.h"
#include "gtest/gtest.h"
using namespace std;

#define LOGPF(format, ...) fprintf(stderr ,"[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

bool NusPCD2Txt(const std::string& pcd_file_path, const std::string& txt_file_path)
{
    if(pcd_file_path.empty() || txt_file_path.empty())
    {
        return false;
    }
    std::ifstream pcd_file(pcd_file_path, std::ios::in | std::ios::binary);
    if (!pcd_file.good()) 
    {
        LOGPF("Error during openning the pcd_file: %s.", pcd_file_path.c_str());
        return false;
    }

    std::ofstream txt_file;
    txt_file.open(txt_file_path , std::ios::out);  

    float max_i = 0;
    int counter = 0;
    while (pcd_file) 
    {
        char tmp_buf[512] = {0};
        float x,y,z,i,r;
        pcd_file.read(reinterpret_cast<char*>(&x), sizeof(float));
        pcd_file.read(reinterpret_cast<char*>(&y), sizeof(float));
        pcd_file.read(reinterpret_cast<char*>(&z), sizeof(float));
        pcd_file.read(reinterpret_cast<char*>(&i), sizeof(float));
        pcd_file.read(reinterpret_cast<char*>(&r), sizeof(float));
        max_i = std::max(max_i, i);
        txt_file << std::scientific << std::setprecision(9) << x << " " << y << " " << z << " " << i << " " << r << std::endl;
        counter ++;
    }
    LOGPF("load pcd_file (%s) points: (%d)", pcd_file_path.c_str(), counter);
    pcd_file.close();
    txt_file.close();

    return true;
}

int Txt2ArrayV2(float** points_array_ptr , string file_name , int num_feature = 4)
{
  ifstream InFile;
  InFile.open(file_name.data());
  assert(InFile.is_open());
  std::vector<float> temp_points;

  std::string line;
  size_t points_counter = 0;
  while(std::getline(InFile, line))
  {
    points_counter ++;
    // LOGPF("line[%d]: %s", points_counter, line.c_str());
    float x, y, z, i, r;
    sscanf(line.c_str(), "%e %e %e %e %e\n", &x, &y, &z, &i, &r);
    // LOGPF("point[%d] x=%f, y=%f, z=%f, i=%f, r=%f", x, y, z, i, r);
    temp_points.push_back(x);
    temp_points.push_back(y);
    temp_points.push_back(z);
    temp_points.push_back(i);
    // temp_points.push_back(r);
    /** r stands for scan-round, 
     *  in 10-sweeps file, it varies from 0.0, 0.05, 0.10, ... 0.45,
     *  so in single sweep file, let r=0 works fine. */
    temp_points.push_back(0.0f); 
  }

  InFile.close();
  LOGPF("temp_points size: %ld", temp_points.size());
  size_t points_array_size = points_counter * num_feature;

  *points_array_ptr = new float[points_array_size];
  float* points_array = *points_array_ptr;
  for(int i=0; i<points_counter; ++i)
  {
      for(int j=0; j<num_feature; j++)
      {
          size_t out_idx = i*num_feature + j;
          size_t tmp_idx = i*5 + j;
          points_array[out_idx] = temp_points[tmp_idx];
      }
  }
  return points_counter;
}

int Txt2Arrary( float* &points_array , string file_name , int num_feature = 4)
{
  ifstream InFile;
  InFile.open(file_name.data());
  assert(InFile.is_open());

  vector<float> temp_points;
  string c;

  uint32_t counter = 0;
  while (!InFile.eof())
  {
      InFile >> c;
      temp_points.push_back(atof(c.c_str()));
      // LOGPF("c: %s, temp_points size: %ld", c.c_str(), temp_points.size());
  }
  LOGPF("temp_points size: %ld", temp_points.size());
  points_array = new float[temp_points.size()];
  for (int i = 0 ; i < temp_points.size() ; ++i) {
    counter += 1;
    // if(counter % 5 == 0)
    // {
    //   continue;
    // }
    points_array[i] = temp_points[i];
  }

  InFile.close();  
  return temp_points.size() / num_feature;
  // printf("Done");
};

void Boxes2Txt( std::vector<float> boxes , std::vector<int> cls, string file_name , int num_feature = 7)
{
    ofstream ofFile;
    ofFile.open(file_name , std::ios::out );  
    if (ofFile.is_open()) {
        for (int i = 0 ; i < boxes.size() / num_feature ; ++i) 
        {
            /** only keep cars */
            // if(cls[i] != 0)
            // {
            //   continue;
            // }
            for (int j = 0 ; j < num_feature ; ++j) 
            {
              // if(j%(num_feature-1) == 0)
              // {
              //   boxes.at(i * num_feature + j) += 1.571;
              // }
              ofFile << boxes.at(i*num_feature+j) << " ";
            }
            ofFile << "\n";
        }
    }
    ofFile.close();
    return ;
};

TEST(PointPillars, __build_model__) {

  // NusPCD2Txt(
  //   "/home/hugoliu/github/nvidia/lidar/PointPillars_MultiHead_40FPS/test/testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.bin",
  //   "/home/hugoliu/github/nvidia/lidar/PointPillars_MultiHead_40FPS/test/testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt"
  // );
  // abort();

  const std::string DB_CONF = "bootstrap.yaml";
  YAML::Node config = YAML::LoadFile(DB_CONF);

  std::string pfe_file,backbone_file; 
  if(config["UseOnnx"].as<bool>()) {
    pfe_file = config["PfeOnnx"].as<std::string>();
    backbone_file = config["BackboneOnnx"].as<std::string>();
  }else {
    pfe_file = config["PfeTrt"].as<std::string>();
    backbone_file = config["BackboneTrt"].as<std::string>();
  }
  std::cout << backbone_file << std::endl;
  const std::string pp_config = config["ModelConfig"].as<std::string>();
  PointPillars pp(
    config["ScoreThreshold"].as<float>(),
    config["NmsOverlapThreshold"].as<float>(),
    config["UseOnnx"].as<bool>(),
    pfe_file,
    backbone_file,
    pp_config
  );
  std::string file_name = config["InputFile"].as<std::string>();
  LOGPF("read points from: %s", file_name.c_str());
  float* points_array;
  int in_num_points;
  // in_num_points = Txt2Arrary(points_array, file_name, 5);
  /** input-feature: x,y,z,i,r  , r is indispensable, let r=0 just works fine */
  in_num_points = Txt2ArrayV2(&points_array, file_name, 5);
  LOGPF("in_num_points: %d", in_num_points);

  for (int i = 0 ; i < 1 ; i++)
  {
    std::vector<float> out_detections;
    std::vector<int> out_labels;
    std::vector<float> out_scores;

    cudaDeviceSynchronize();
    pp.DoInference(points_array, in_num_points, &out_detections, &out_labels , &out_scores);
    cudaDeviceSynchronize();
    int BoxFeature = 7;
    int num_objects = out_detections.size() / BoxFeature;

    std::string boxes_file_name = config["OutputFile"].as<std::string>();
    Boxes2Txt(out_detections, out_labels, boxes_file_name);

    LOGPF("test [%d], num_objects: %d, boxes_file_name: %s", i, num_objects, boxes_file_name.c_str());
    for(int j=0; j<num_objects; j++)
    {
      LOGPF("obj[%d] wlh(%.2f, %.2f, %.2f), label: %d, score: %.2f", \
            j, out_detections[j*BoxFeature+3], out_detections[j*BoxFeature+4], out_detections[j*BoxFeature+5], \
            out_labels[j], out_scores[j]);
    }
    // EXPECT_EQ(num_objects,226);
  }


};
