/********************************************************************************
** Form generated from reading UI file 'detect_3d.ui'
**
** Created by: Qt User Interface Compiler version 5.12.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DETECT_3D_H
#define UI_DETECT_3D_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_detect_3d
{
public:
    QAction *actionexit;
    QWidget *centralWidget;
    QPushButton *point_cloud;
    QTextBrowser *state;
    QPushButton *epipolar;
    QGroupBox *groupBox;
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout;
    QSpinBox *scale;
    QLabel *label_3;
    QSpinBox *frame;
    QLabel *label_2;
    QLabel *label_5;
    QLineEdit *pose_save;
    QCheckBox *save_pose;
    QPushButton *camera_pose;
    QGroupBox *groupBox_2;
    QWidget *gridLayoutWidget_2;
    QGridLayout *gridLayout_2;
    QLabel *label_4;
    QLineEdit *save_path;
    QCheckBox *save;
    QLabel *label;
    QDoubleSpinBox *y_bias;
    QPushButton *distance;
    QPushButton *fast_track;
    QCheckBox *save_iou;
    QPushButton *depth_data;
    QPushButton *depth_filter;
    QMenuBar *menuBar;
    QMenu *menuExit;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *detect_3d)
    {
        if (detect_3d->objectName().isEmpty())
            detect_3d->setObjectName(QString::fromUtf8("detect_3d"));
        detect_3d->resize(1182, 505);
        actionexit = new QAction(detect_3d);
        actionexit->setObjectName(QString::fromUtf8("actionexit"));
        centralWidget = new QWidget(detect_3d);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        point_cloud = new QPushButton(centralWidget);
        point_cloud->setObjectName(QString::fromUtf8("point_cloud"));
        point_cloud->setGeometry(QRect(10, 140, 341, 23));
        state = new QTextBrowser(centralWidget);
        state->setObjectName(QString::fromUtf8("state"));
        state->setGeometry(QRect(10, 200, 1001, 241));
        epipolar = new QPushButton(centralWidget);
        epipolar->setObjectName(QString::fromUtf8("epipolar"));
        epipolar->setGeometry(QRect(10, 170, 341, 20));
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(360, 30, 331, 161));
        gridLayoutWidget = new QWidget(groupBox);
        gridLayoutWidget->setObjectName(QString::fromUtf8("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(10, 19, 311, 141));
        gridLayout = new QGridLayout(gridLayoutWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        scale = new QSpinBox(gridLayoutWidget);
        scale->setObjectName(QString::fromUtf8("scale"));
        scale->setMinimum(1);
        scale->setMaximum(200);
        scale->setValue(10);

        gridLayout->addWidget(scale, 1, 1, 1, 2);

        label_3 = new QLabel(gridLayoutWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        frame = new QSpinBox(gridLayoutWidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setMinimum(1);
        frame->setMaximum(50);
        frame->setValue(5);

        gridLayout->addWidget(frame, 2, 1, 1, 2);

        label_2 = new QLabel(gridLayoutWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        label_5 = new QLabel(gridLayoutWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 0, 0, 1, 1);

        pose_save = new QLineEdit(gridLayoutWidget);
        pose_save->setObjectName(QString::fromUtf8("pose_save"));

        gridLayout->addWidget(pose_save, 0, 1, 1, 1);

        save_pose = new QCheckBox(gridLayoutWidget);
        save_pose->setObjectName(QString::fromUtf8("save_pose"));

        gridLayout->addWidget(save_pose, 0, 2, 1, 1);

        camera_pose = new QPushButton(gridLayoutWidget);
        camera_pose->setObjectName(QString::fromUtf8("camera_pose"));

        gridLayout->addWidget(camera_pose, 3, 0, 1, 3);

        groupBox_2 = new QGroupBox(centralWidget);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 30, 341, 101));
        gridLayoutWidget_2 = new QWidget(groupBox_2);
        gridLayoutWidget_2->setObjectName(QString::fromUtf8("gridLayoutWidget_2"));
        gridLayoutWidget_2->setGeometry(QRect(10, 19, 321, 80));
        gridLayout_2 = new QGridLayout(gridLayoutWidget_2);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        label_4 = new QLabel(gridLayoutWidget_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_2->addWidget(label_4, 1, 0, 1, 1);

        save_path = new QLineEdit(gridLayoutWidget_2);
        save_path->setObjectName(QString::fromUtf8("save_path"));

        gridLayout_2->addWidget(save_path, 0, 1, 1, 1);

        save = new QCheckBox(gridLayoutWidget_2);
        save->setObjectName(QString::fromUtf8("save"));

        gridLayout_2->addWidget(save, 0, 2, 1, 1);

        label = new QLabel(gridLayoutWidget_2);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout_2->addWidget(label, 0, 0, 1, 1);

        y_bias = new QDoubleSpinBox(gridLayoutWidget_2);
        y_bias->setObjectName(QString::fromUtf8("y_bias"));
        y_bias->setValue(5.000000000000000);

        gridLayout_2->addWidget(y_bias, 1, 1, 1, 2);

        distance = new QPushButton(gridLayoutWidget_2);
        distance->setObjectName(QString::fromUtf8("distance"));

        gridLayout_2->addWidget(distance, 2, 0, 1, 3);

        fast_track = new QPushButton(centralWidget);
        fast_track->setObjectName(QString::fromUtf8("fast_track"));
        fast_track->setGeometry(QRect(700, 90, 151, 20));
        save_iou = new QCheckBox(centralWidget);
        save_iou->setObjectName(QString::fromUtf8("save_iou"));
        save_iou->setGeometry(QRect(700, 68, 141, 20));
        depth_data = new QPushButton(centralWidget);
        depth_data->setObjectName(QString::fromUtf8("depth_data"));
        depth_data->setGeometry(QRect(700, 120, 151, 19));
        depth_filter = new QPushButton(centralWidget);
        depth_filter->setObjectName(QString::fromUtf8("depth_filter"));
        depth_filter->setGeometry(QRect(700, 150, 151, 20));
        detect_3d->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(detect_3d);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1182, 20));
        menuExit = new QMenu(menuBar);
        menuExit->setObjectName(QString::fromUtf8("menuExit"));
        detect_3d->setMenuBar(menuBar);
        mainToolBar = new QToolBar(detect_3d);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        detect_3d->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(detect_3d);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        detect_3d->setStatusBar(statusBar);

        menuBar->addAction(menuExit->menuAction());
        menuExit->addAction(actionexit);

        retranslateUi(detect_3d);

        QMetaObject::connectSlotsByName(detect_3d);
    } // setupUi

    void retranslateUi(QMainWindow *detect_3d)
    {
        detect_3d->setWindowTitle(QApplication::translate("detect_3d", "detect_3d", nullptr));
        actionexit->setText(QApplication::translate("detect_3d", "exit", nullptr));
        point_cloud->setText(QApplication::translate("detect_3d", "Draw Point cloud", nullptr));
        epipolar->setText(QApplication::translate("detect_3d", "Epipolar Geometry", nullptr));
        groupBox->setTitle(QApplication::translate("detect_3d", "Camera Pose", nullptr));
        label_3->setText(QApplication::translate("detect_3d", "Frame", nullptr));
        label_2->setText(QApplication::translate("detect_3d", "Scale", nullptr));
        label_5->setText(QApplication::translate("detect_3d", "Save Path: ", nullptr));
        save_pose->setText(QApplication::translate("detect_3d", "Save", nullptr));
        camera_pose->setText(QApplication::translate("detect_3d", "Camera pose", nullptr));
        groupBox_2->setTitle(QApplication::translate("detect_3d", "Get Distance", nullptr));
        label_4->setText(QApplication::translate("detect_3d", "ybias", nullptr));
        save->setText(QApplication::translate("detect_3d", "Save", nullptr));
        label->setText(QApplication::translate("detect_3d", "Distance Save Path:", nullptr));
        distance->setText(QApplication::translate("detect_3d", "Get Distance", nullptr));
        fast_track->setText(QApplication::translate("detect_3d", "Faster Tracking", nullptr));
        save_iou->setText(QApplication::translate("detect_3d", "Save IOU Info.", nullptr));
        depth_data->setText(QApplication::translate("detect_3d", "Track with Depth data", nullptr));
        depth_filter->setText(QApplication::translate("detect_3d", "Depth Filter", nullptr));
        menuExit->setTitle(QApplication::translate("detect_3d", "Exit", nullptr));
    } // retranslateUi

};

namespace Ui {
    class detect_3d: public Ui_detect_3d {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DETECT_3D_H
