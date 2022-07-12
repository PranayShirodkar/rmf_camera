#ifndef BOUNDINGBOX3D_HPP
#define BOUNDINGBOX3D_HPP

#include <iostream>

struct BoundingBox3D {
    std::string classification;
    int id;
    cv::Point3d position;
    cv::Vec3d size;

    std::string toString() const;
};

std::ostream& operator << (std::ostream &os, const BoundingBox3D &bb) {
    return (os << "Class: " << bb.classification << " Id: " << bb.id
    << " Position: " << bb.position.x << " " << bb.position.y << " " << bb.position.z
    << " Size: " << bb.size[0] << " " << bb.size[1] << " " << bb.size[2] << std::endl);
}

std::string BoundingBox3D::toString() const {
    std::stringstream ss;
    ss << (*this);
    return ss.str();
}


#endif