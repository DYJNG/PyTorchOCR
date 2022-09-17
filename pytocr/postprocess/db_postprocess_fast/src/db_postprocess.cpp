#include "db_postprocess.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

using namespace std;
using namespace cv;

namespace py = pybind11;


namespace db_postprocess {

void DBPostProcessor::GetContourArea(const std::vector<std::vector<float>> &box, 
                                     float unclip_ratio, float &distance) {
	int pts_num = 4;
	float area = 0.0f;
	float dist = 0.0f;
	for (int i = 0; i < pts_num; i++) {
		area += box[i][0] * box[(i + 1) % pts_num][1] -
				box[i][1] * box[(i + 1) % pts_num][0];
		dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
					  (box[i][0] - box[(i + 1) % pts_num][0]) + 
					  (box[i][1] - box[(i + 1) % pts_num][1]) *
					  (box[i][1] - box[(i + 1) % pts_num][1]));
	}
	area = fabs(float(area / 2.0));

	distance = area * unclip_ratio / dist;
}

cv::RotatedRect DBPostProcessor::UnClip(std::vector<std::vector<float>> box, 
                                        const float &unclip_ratio) {
	float distance = 1.0;

	GetContourArea(box, unclip_ratio, distance);

	ClipperLib::ClipperOffset offset;
	ClipperLib::Path p;
	p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
	  << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
	  << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
	  << ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
	offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

	ClipperLib::Paths soln;
	offset.Execute(soln, distance);
	std::vector<cv::Point2f> points;

	for (int j = 0; j < soln.size(); j++) {
		for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
			points.emplace_back(soln[j][i].X, soln[j][i].Y);
		}
	}
	cv::RotatedRect res;
	if (points.size() <= 0) {
		res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
	} else {
		res = cv::minAreaRect(points);
	}
	return res;
}

float **DBPostProcessor::Mat2Vec(cv::Mat mat) {
	auto **array = new float *[mat.rows];
	for (int i = 0; i < mat.rows; ++i)
		array[i] = new float[mat.cols];
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			array[i][j] = mat.at<float>(i, j);
		}
	}
	return array;
}

std::vector<std::vector<int>> 
DBPostProcessor::OrderPointsClockwise(std::vector<std::vector<int>> pts) {
	std::vector<std::vector<int>> box = pts;
	std::sort(box.begin(), box.end(), XsortInt);

	std::vector<std::vector<int>> leftmost = {box[0], box[1]};
	std::vector<std::vector<int>> rightmost = {box[2], box[3]};

	if (leftmost[0][1] > leftmost[1][1])
		std::swap(leftmost[0], leftmost[1]);
	
	if (rightmost[0][1] > rightmost[1][1])
		std::swap(rightmost[0], rightmost[1]);
	
	std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], 
										  rightmost[1], leftmost[1]};
	return rect;
}

std::vector<std::vector<float>> DBPostProcessor::Mat2Vector(cv::Mat mat) {
	std::vector<std::vector<float>> img_vec;
	std::vector<float> tmp;

	for (int i = 0; i < mat.rows; ++i) {
		tmp.clear();
		for (int j = 0; j < mat.cols; ++j) {
			tmp.push_back(mat.at<float>(i, j));
		}
		img_vec.push_back(tmp);
	}
	return img_vec;
}

cv::Mat DBPostProcessor::get_affine_transform(const cv::Point2f &center, 
											  const float img_maxsize, 
											  const float target_size, 
											  const int inv) {
	cv::Point2f srcTriangle[3];
	cv::Point2f dstTriangle[3];

	srcTriangle[0] = center;
	srcTriangle[1] = center + cv::Point2f(0, img_maxsize / 2.0);
	dstTriangle[0] = cv::Point2f(target_size / 2.0, target_size / 2.0);
	dstTriangle[1] = dstTriangle[0] + cv::Point2f(0, target_size / 2.0);
	dstTriangle[2] = cv::Point2f(0, 0);

	if (center.x >= center.y)
		srcTriangle[2] = cv::Point2f(0, center.y - center.x);
	else
		srcTriangle[2] = cv::Point2f(center.x - center.y, 0);

	cv::Mat affineMat(2, 3, CV_32FC1);
	if (inv == 0)
		affineMat = cv::getAffineTransform(srcTriangle, dstTriangle);
	else
		affineMat = cv::getAffineTransform(dstTriangle, srcTriangle);
	return affineMat;
}

cv::Point2f DBPostProcessor::transform_preds(const cv::Mat&warpMat, 
											 const cv::Point2f& pt) {
	cv::Mat pt_mid(1, 3, CV_64FC1);
	pt_mid.at<double>(0, 0) = pt.x;
	pt_mid.at<double>(0, 1) = pt.y;
	pt_mid.at<double>(0, 2) = 1.0;
	cv::Mat new_pt = pt_mid * warpMat;
	return cv::Point2f(new_pt.at<double>(0, 0), new_pt.at<double>(0, 1));
}

bool DBPostProcessor::XsortFp32(std::vector<float> a, std::vector<float> b) {
	if (a[0] != b[0])
		return a[0] < b[0];
	return false;
}

bool DBPostProcessor::XsortInt(std::vector<int> a, std::vector<int> b) {
	if (a[0] != b[0])
		return a[0] < b[0];
	return false;
}

std::vector<std::vector<float>> DBPostProcessor::GetMiniBoxes(cv::RotatedRect box, 
                                                 float &ssid) {
	ssid = std::max(box.size.width, box.size.height);

	cv::Mat points;
	cv::boxPoints(box, points);
	
	auto array = Mat2Vector(points);
	std::sort(array.begin(), array.end(), XsortFp32);

	std::vector<float> idx1 = array[0], idx2 = array[1], 
					   idx3 = array[2], idx4 = array[3];
	if (array[3][1] <= array[2][1]) {
		idx2 = array[3];
		idx3 = array[2];
	} else {
		idx2 = array[2];
		idx3 = array[3];
	}
	if (array[1][1] <= array[0][1]) {
		idx1 = array[1];
		idx4 = array[0];
	} else {
		idx1 = array[0];
		idx4 = array[1];
	}

	array[0] = idx1;
	array[1] = idx2;
	array[2] = idx3;
	array[3] = idx4;

	return array;
}

float DBPostProcessor::BoxScore(std::vector<cv::Point> contour, 
								cv::Mat pred) {
	std::vector<cv::Point> poly = contour;
	int width = pred.cols;
	int height = pred.rows;
	
	int xmin = width, xmax = -1, ymin = height, ymax = -1;
	for (int i = 0; i < poly.size(); i++) {
		xmin = xmin > poly[i].x ? poly[i].x : xmin;
		xmax = xmax < poly[i].x ? poly[i].x : xmax;
		ymin = ymin > poly[i].y ? poly[i].y	: ymin;
		ymax = ymax < poly[i].y	? poly[i].y : ymax;
	}
	xmax = (std::min)((std::max)(xmax, 0), width - 1);
	xmin = (std::max)((std::min)(xmin, width - 1), 0);
	ymax = (std::min)((std::max)(ymax, 0), height - 1);
	ymin = (std::max)((std::min)(ymin, height - 1), 0);

	for (int i = 0; i < poly.size(); i++) {
		poly[i].x -= xmin;
		poly[i].y -= ymin;
	}
	
	std::vector<std::vector<cv::Point>> pts;
	pts.push_back(poly);

	cv::Mat mask;
  	mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
	cv::fillPoly(mask, pts, cv::Scalar(1), 1);

	cv::Mat croppedImg;
	pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
		.copyTo(croppedImg);
	auto score = cv::mean(croppedImg, mask)[0];
	return score;
}

std::vector<std::vector<std::vector<int>>> 
DBPostProcessor::BoxesFromBitmap(const cv::Mat pred, 
								 const cv::Mat bitmap, 
								 const float &box_thresh, 
								 const float &det_db_unclip_ratio,
								 int src_w, int src_h,
								 bool use_padding_resize) {
	const int min_size = 3;
	const int max_candidates = 1000;

	int width = bitmap.cols;
	int height = bitmap.rows;

	std::vector<std::vector<cv::Point>> contours;

	cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	int num_contours =
		contours.size() >= max_candidates ? max_candidates : contours.size();

	std::vector<std::vector<std::vector<int>>> boxes;
	std::vector<float> scores;

	for (int _i = 0; _i < num_contours; _i++) {
		if (contours[_i].size() <= 2) {
			continue;
		}
		float ssid;
		cv::RotatedRect box = cv::minAreaRect(contours[_i]);
		auto array = GetMiniBoxes(box, ssid);

		auto box_for_unclip = array;
		// end get_mini_box

		if (ssid < min_size) {
			continue;
		}

		float score;
		score = BoxScore(contours[_i], pred);
		//     cout<<score<<endl;
		if (score < box_thresh)
			continue;

		// start for unclip
		cv::RotatedRect points = UnClip(box_for_unclip, det_db_unclip_ratio);
		if (points.size.height < 1.001 && points.size.width < 1.001) {
			continue;
		}
		// end for unclip

		cv::RotatedRect clipbox = points;
		auto cliparray = GetMiniBoxes(clipbox, ssid);

		if (ssid < min_size + 2)
			continue;

		std::vector<std::vector<int>> intcliparray;

		if (use_padding_resize) {
			cv::Point2f center(src_w / 2.0, src_h / 2.0);
			int img_maxsize = src_w > src_h ? src_w : src_h;
			int square_size = height;
			cv::Mat warpMat = get_affine_transform(center, img_maxsize, square_size, 1).t();
			for (int j = 0; j < 4; j++) {
				cv::Point2f tmp_pt(cliparray[j][0], cliparray[j][1]);
				tmp_pt = transform_preds(warpMat, tmp_pt);
				std::vector<int> a{int(clampf(roundf(tmp_pt.x), 0, float(src_w))),
								   int(clampf(roundf(tmp_pt.y), 0, float(src_h)))};
				intcliparray.push_back(a);
			}
		}
		else {
			for (int j = 0; j < 4; j++) {
				std::vector<int> a{int(clampf(roundf(cliparray[j][0] / float(width) * 
									          float(src_w)), 0, float(src_w))),
								   int(clampf(roundf(cliparray[j][1] / float(height) * 
									          float(src_h)), 0, float(src_h)))};
				intcliparray.push_back(a);
			}
		}
		boxes.push_back(intcliparray);
		scores.push_back(score);  // how to return : struct?
	} // end for

	return boxes;
} 

std::vector<std::vector<std::vector<int>>> DBProcess(py::array_t<float, py::array::c_style> pred,
													 py::array_t<uint8_t, py::array::c_style> bitmap,
													 float box_thresh,
													 float det_db_unclip_ratio,
													 int src_w, int src_h,
								 					 bool use_padding_resize) {
    auto buf_pred = pred.request();
    auto buf_bitmap= bitmap.request();
    
    auto ptr_pred = static_cast<float *>(buf_pred.ptr);
    auto ptr_bitmap = static_cast<uint8_t *>(buf_bitmap.ptr);
    
    cv::Mat pred_mat;
    cv::Mat bitmap_mat;
    
    vector<long int> data_shape = buf_pred.shape;
    
    std::vector<std::vector<std::vector<int>>> boxes;
    
    pred_mat = Mat::zeros(data_shape[0], data_shape[1], CV_32FC1);
    for (int x = 0; x < pred_mat.rows; ++x) {
        for (int y = 0; y < pred_mat.cols; ++y) {
            pred_mat.at<float>(x, y) = ptr_pred[x * data_shape[1] + y];
        }
	}
    bitmap_mat = Mat::zeros(data_shape[0], data_shape[1], CV_8UC1);
    for (int x = 0; x < bitmap_mat.rows; ++x) {
        for (int y = 0; y < bitmap_mat.cols; ++y) {
            bitmap_mat.at<char>(x, y) = ptr_bitmap[x * data_shape[1] + y];
        }
	}
    DBPostProcessor *db_post_processor_ = new DBPostProcessor();
    boxes = db_post_processor_->BoxesFromBitmap(pred_mat, bitmap_mat, 
												box_thresh, 
												det_db_unclip_ratio,
												src_w, src_h,
												use_padding_resize);
    delete db_post_processor_;
    return boxes;
}

}  // namespace db_postprocess

PYBIND11_MODULE(db_postprocess, m){
    m.def("db_postprocess", &db_postprocess::DBProcess, 
		  "re-implementation db postprocess algorithm(cpp)", 
		  py::arg("pred"), py::arg("bitmap"), 
		  py::arg("box_thresh"), 
		  py::arg("det_db_unclip_ratio"),
		  py::arg("src_w"), py::arg("src_h"),
		  py::arg("use_padding_resize"));
}
