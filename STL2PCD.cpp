//Author: Khuong Thanh Gia Hieu
//Bach Khoa University - CK16KSCD

#include "HPR.h"
#include "meshSampling.h"
#include "pcl/visualization/cloud_viewer.h"
#include <thread>
#include <mutex>

int main()
{
	//In this project, we use Hidden Point Removal method from this paper:
	//Direct Visibility of Point Sets
    //Sagi Katz, Ayellet Tal, Ronen Basri
	//2007

	//First, sample the hold mesh using function meshSampling
	std::string model_filename_ = "../data/Tshape.STL";
    std::cout << "Loading mesh..." << std::endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_sampling(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>());
    meshSampling(model_filename_, 1000000, 0.001f, false, model_sampling);

    //---- Calculate point cloud from 6 views and combine ------
    std::vector<std::vector<float>> camera_pos(6);
    pcl::PointXYZ minPt, maxPt, avgPt;

    pcl::getMinMax3D(*model_sampling, minPt, maxPt);
    avgPt.x = (minPt.x + maxPt.x) / 2;
    avgPt.y = (minPt.y + maxPt.y) / 2;
    avgPt.z = (minPt.z + maxPt.z) / 2;

    float cube_length = std::max(maxPt.x - minPt.x, std::max(maxPt.y - minPt.y, maxPt.z - minPt.z));

    minPt.x = avgPt.x - cube_length;
    minPt.y = avgPt.y - cube_length;
    minPt.z = avgPt.z - cube_length;
    maxPt.x = avgPt.x + cube_length;
    maxPt.y = avgPt.y + cube_length;
    maxPt.z = avgPt.z + cube_length;

    camera_pos[0] = { avgPt.x, minPt.y, avgPt.z };
    camera_pos[1] = { maxPt.x, avgPt.y, avgPt.z };
    camera_pos[2] = { avgPt.x, maxPt.y, avgPt.z };
    camera_pos[3] = { minPt.x, avgPt.y, avgPt.z };
    camera_pos[4] = { avgPt.x, avgPt.y, maxPt.z };
    camera_pos[5] = { avgPt.x, avgPt.y, minPt.z };


    for (int i = 0; i < static_cast<int>(camera_pos.size()); ++i)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_HPR(new pcl::PointCloud<pcl::PointXYZ>());
        HPR(model_sampling, camera_pos[i], 3, cloud_xyz_HPR); //calculate from i view with HPR

        *model += *cloud_xyz_HPR;//combine

    }

	// ------- centering the model ----------------
	Eigen::Vector3d sum_of_pos = Eigen::Vector3d::Zero();
	for (const auto& p : *(model)) sum_of_pos += p.getVector3fMap().cast<double>();

	Eigen::Matrix4d transform_centering = Eigen::Matrix4d::Identity();
	transform_centering.topRightCorner<3, 1>() = -sum_of_pos / model->size();

	pcl::transformPointCloud(*model, *model, transform_centering);
	pcl::transformPointCloud(*model, *model, Eigen::Vector3f(0, 0, 0), Eigen::Quaternionf(0.7071, 0, -0.7071, 0));

	std::cout << "Done Preparing Model....." << std::endl;


	// --------- save the file -------------
	std::string output_model_filename_ = "../data/Tshape.pcd";
	pcl::io::savePCDFileASCII (output_model_filename_, *model);
	std::cout << "Saved " << model->size () << " data points to " << output_model_filename_ << std::endl;

	// ----------- Visualize sampled cloud --------------
	pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
	viewer.showCloud (model);
	while (!viewer.wasStopped());

	return 0;
}