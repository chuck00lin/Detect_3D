#include "detect_3d.hpp"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    detect_3d w;
    w.show();

    return a.exec();
}
