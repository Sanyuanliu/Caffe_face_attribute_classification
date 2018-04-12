// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
DEFINE_int32(label_num, 1,
    "Optional: How many numbers should we encode the label.");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 5) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset_multi_labels");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  const int label_num = FLAGS_label_num;

  std::ifstream infile(argv[2]);
  if (!infile.good()) {
    std::cout<<"Can not open: "<<argv[2]<<std::endl;
    return -1;
  }

  std::vector<std::pair<std::string, vector<float> > > lines;
  vector<float> labels(label_num);
 
  std::string line;
  
/*size_t pos;

  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines.push_back(std::make_pair(line.substr(0, pos), label));
  }
*/
  std::string filename;
  while (std::getline(infile, line)) {
    //std::cout<<line<<std::endl;
    std::stringstream ss(line);
    ss >> filename;
    //std::cout<<filename;
    for (int i=0; i<label_num; ++i) {
      ss >> labels[i];
      //std::cout<<labels[i];
    }
    ss.clear();
    //std::cout<<std::endl;
    lines.push_back(std::make_pair(filename, labels));

  }

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db_image(db::GetDB(FLAGS_backend));
  scoped_ptr<db::DB> db_labels(db::GetDB(FLAGS_backend));
  db_image->Open(argv[3], db::NEW);
  db_labels->Open(argv[4], db::NEW);
  scoped_ptr<db::Transaction> txn_image(db_image->NewTransaction());
  scoped_ptr<db::Transaction> txn_labels(db_labels->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum_image;
  Datum datum_labels;
  int count = 0;
  int data_size_image = 0;
  int data_size_labels = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadImageToDatum(root_folder + lines[line_id].first,
        lines[line_id].second[0], resize_height, resize_width, is_color,
        enc, &datum_image);
    if (status == false) continue;

    datum_labels.set_height(1);
    datum_labels.set_width(1);
    datum_labels.set_channels(label_num);

    for (int index_label = 0; index_label < lines[line_id].second.size(); index_label++)
    {
      float tmp_float_value = lines[line_id].second[index_label];
      //std::cout<<tmp_float_value<<" ";
      datum_labels.add_float_data(tmp_float_value);
    }
    //int count_tmp = datum_labels.float_data_size();
    //std::cout<<"line_id: "<<line_id<<" count_tmp: "<<count_tmp<<std::endl;
    //std::cout<<std::endl;
    if (check_size) {
      if (!data_size_initialized) {
        data_size_image = datum_image.channels() * datum_image.height() * datum_image.width();
        data_size_labels = datum_labels.channels() * datum_labels.height() * datum_labels.width();
        data_size_initialized = true;
      } else {
        const std::string& data_image = datum_image.data();
        CHECK_EQ(data_image.size(), data_size_image) << "Incorrect data field size "
            << data_image.size();
        //todo: Here can not be checked correctly. 
        const std::string& data_labels = datum_labels.data();
        CHECK_EQ(data_labels.size(), data_size_labels) << "Incorrect data field size "
            << data_labels.size();
      }
    }
    // sequential
    string key_str_image = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;
    string key_str_labels = caffe::format_int(line_id, 8) + "labels_" + lines[line_id].first;

    // Put in db
    string out_image;
    string out_labels;
    CHECK(datum_image.SerializeToString(&out_image));
    CHECK(datum_labels.SerializeToString(&out_labels));

    datum_labels.clear_float_data();
    txn_image->Put(key_str_image, out_image);
    txn_labels->Put(key_str_labels, out_labels);

    if (++count % 1000 == 0) {
      // Commit db
      txn_image->Commit();
      txn_labels->Commit();

      txn_image.reset(db_image->NewTransaction());
      txn_labels.reset(db_labels->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn_image->Commit();
    txn_labels->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
