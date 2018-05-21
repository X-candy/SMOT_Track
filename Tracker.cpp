#include "Tracker.h"


CTracker::CTracker(void)
{
    m_dSimilarity_method = 1;
    m_nHor = 80;
    m_ratio_threhold = 3;
    m_dMin_similarity = 0.01;
    m_dMota_threhold = 0.5;
    m_dEta_max = 3;
    m_bDebug = 1;
    m_bQcheck = false;
    m_dGap = 0;
    m_dHor_max = INF;
    m_dGap_max = INF;
    m_dSlope_max = INF;
    m_lastminD2 = 0;

    m_nHistoryLen = 15;
    m_detect_rect_squence.resize(m_nHistoryLen);
    m_process_frameNumCount = 0;
    //m_B.resize(m_nHistoryLen);
    //m_distanceSQ.resize(m_nHistoryLen);
	m_ObjCount =0;


    temp_c = 1;
}


CTracker::~CTracker(void)
{
}

double CTracker::CalSimilar(Mat src, Mat src1)
{
	double	feature_similar=0;
	Mat res;
	matchTemplate( src,src1, res, CV_TM_CCOEFF_NORMED);    
	feature_similar  = ((float *)(res.data))[0];
	return  feature_similar;
}

int CTracker::CheckDetectResult(Mat _frame,Mat _bgRgb,Mat _gmmMask, Mat _detectMask, Mat _bgSobel,Mat _frameSobel, Mat _diff3)
{
	//获取成功推入样本的均值
	double drcSimilarAvg=0;
	double drcSobelSimilarAvg=0;
	int    nCount=0;
	for (int i=0;i<m_itl.size();i++)
	{
		if (m_itl[i].t_end == m_process_frameNumCount)
		{
			int  nLast = m_itl[i].rect_data.size()-1;
			Rect _Rc=m_itl[i].rect_data[nLast];
			Mat _frameRc(m_frame,_Rc);
			Mat _bgRc(_bgRgb,_Rc);
			Mat _frameSobelRc(_frameSobel,_Rc);
			Mat _bgSobelRc(_bgSobel,_Rc);
			drcSimilarAvg+=CalSimilar(_frameRc,_bgRc);
			drcSobelSimilarAvg+=CalSimilar(_frameSobelRc,_bgSobelRc);
			nCount++;
			m_itl[i].bNeedShow=false;
		}
	}
	drcSimilarAvg=drcSimilarAvg/(nCount+0.01);
	drcSobelSimilarAvg=drcSobelSimilarAvg/(nCount+0.01);

	//未成功推入样本的校验
	for (int i=0;i<m_itl.size();i++)
	{
		if(m_itl[i].t_end != m_process_frameNumCount)
		{
			int  nLast = m_itl[i].rect_data.size()-1;
			Rect _Rc=m_itl[i].rect_data[nLast];
			Mat _frameRc(m_frame,_Rc);
			Mat _bgRc(_bgRgb,_Rc);
			Mat _frameSobelRc(_frameSobel,_Rc);
			Mat _bgSobelRc(_bgSobel,_Rc);

			double drcSimilar      = CalSimilar(_frameRc,_bgRc);
			double drcSobelSimilar = CalSimilar(_frameSobelRc,_bgSobelRc);
			if (drcSimilar<drcSimilarAvg*0.9||drcSobelSimilar<drcSobelSimilarAvg*0.9)
			{
				m_itl[i].bNeedShow=true;
			}
			else
			{
				m_itl[i].bNeedShow=false;
			}
		}
	}
	return 0;
}

int CTracker::tracker(int _frame_num, vector<Rect> _detect_rect, Mat _frame,Mat _gmmMask, Mat _detectMask, Mat _bgSobel,Mat _frameSobel, Mat _diff3,Mat _bgRgb)
{
    _frame.copyTo(m_frame);
#if TRACK_DEBUG
	printf("\n*******************%d\n", m_process_frameNumCount);

    for(int i = 0; i < _detect_rect.size(); i++)
    {
		Rect temp_rect;
		temp_rect.x =  _detect_rect[i].x *2 ;
		temp_rect.y =  _detect_rect[i].y *2 ;
		temp_rect.width =  _detect_rect[i].width *2 ;
		temp_rect.height =  _detect_rect[i].height *2 ;
        rectangle(m_frame, temp_rect, Scalar(255, 0, 0));
    }
#endif
#if TRACK_DEBUG
	printf("InputDetectRect start\n");
#endif
    InputDetectRect(m_detect_rect_squence, m_process_frameNumCount, _detect_rect);
#if TRACK_DEBUG
	printf("InputDetectRect end\n");
	printf("FindAssociations start\n");
#endif
    FindAssociations(m_detect_rect_squence, m_ratio_threhold, m_B, m_distanceSQ);
#if TRACK_DEBUG
	printf("FindAssociations end\n");
    printf("LinkDetectionTracklets start\n");
#endif
	LinkDetectionTracklets(m_detect_rect_squence, m_B, m_distanceSQ, m_itl);
#if TRACK_DEBUG
	printf("LinkDetectionTracklets end\n");
#endif
	CheckDetectResult( m_frame,_bgRgb,_gmmMask,_detectMask,_bgSobel, _frameSobel,_diff3);
    int null_count = 0;
    for(int i = 0; i < m_detect_rect_squence.size(); i++)
    {
        if(m_detect_rect_squence[i].detect_rect.size() == 0)
        {
            null_count ++;
        }
    }
    if(null_count == m_detect_rect_squence.size() )
    {
        m_itl.clear();
    }

  	//去除过久的跟踪线
	int N_itlh = m_itl.size();
	if(N_itlh>0)
	{
		int i = 0;
		while(i < N_itlh)
		{
			
			if(m_itl[i].t_end < m_process_frameNumCount - m_nHistoryLen*3 &&  m_process_frameNumCount - m_nHistoryLen*3 >0)
			{
				if(m_itl[i].t_end  == m_process_frameNumCount)
					break;
				m_itl.erase(m_itl.begin() + i);
				N_itlh--;
				continue;
			}
			i++;
		}
	}

#if TRACK_DEBUG
	printf("Associate_ITL start\n");
#endif
    Associate_ITL(m_itl);
#if TRACK_DEBUG
	printf("Associate_ITL end\n");
#endif
	vector<int> erase_map;
	erase_map.clear();
	if(m_MerageMap.size()>0)
	{
		int N_MerageMap = m_MerageMap.size();
		if(N_MerageMap>0)
		{
			int merageIndx0=0;
			int merageIndx1=0;
			for(int i=0;i<N_MerageMap;i++)
			{
				for(int k=0;k<m_itl.size();k++)
				{
					if(m_itl[k].id == m_MerageMap[i][0])
					{
						merageIndx0 = k;
						break;
					}
				}

				for(int k=0;k<m_itl.size();k++)
				{
					if(m_itl[k].id == m_MerageMap[i][1])
					{
						merageIndx1 = k;
						break;
					}
				}
#if TRACK_DEBUG
				//printf("合并序列 start\n");
#endif
				
				//合并序列
				if(m_itl[merageIndx0].t_end  <= m_itl[merageIndx1].t_start)
				{
					int gap = m_itl[merageIndx1].t_start - m_itl[merageIndx0].t_end;
					if(gap < 0 )
						cout<<"Error"<<endl;
					m_itl[merageIndx0].t_end = m_itl[merageIndx1].t_end;
					//	m_itl[merageIndx0].length = m_itl[merageIndx0].t_end -m_itl[merageIndx0].t_start + 1;
					Mat tempMat = Mat();
					Mat XY12_data = Mat();
					tempMat = m_itl[merageIndx0].xy_data.t();
					XY12_data.push_back(tempMat);
					tempMat = Mat::zeros(gap,2, CV_32F);
					XY12_data.push_back(tempMat);
					tempMat = m_itl[merageIndx1].xy_data.t();
					if(gap == 0)
						tempMat = tempMat.rowRange(1,tempMat.rows);

					XY12_data.push_back(tempMat);
					XY12_data = XY12_data.t();

					XY12_data.copyTo(m_itl[merageIndx0].xy_data);
					//cout<<XY12_data<<endl;
					m_itl[merageIndx0].length= XY12_data.cols;

					Mat tempOmegaMat = Mat();
					Mat Omega_data = Mat();
					tempOmegaMat = m_itl[merageIndx0].omega.t();
					Omega_data.push_back(tempOmegaMat);
					tempOmegaMat = Mat::zeros(gap,1, CV_8U);
					Omega_data.push_back(tempOmegaMat);
					tempOmegaMat = m_itl[merageIndx1].omega.t();
					if(gap == 0)
						tempOmegaMat = tempOmegaMat.rowRange(1,tempOmegaMat.rows);
					Omega_data.push_back(tempOmegaMat);
					Omega_data = Omega_data.t();

					Omega_data.copyTo(m_itl[merageIndx0].omega);
					/*	cout<<Omega_data<<endl;*/

					Rect temp_rect=Rect();

					for(int k=0;k<gap;k++)
					{
						m_itl[merageIndx0].rect_data.push_back(temp_rect);
					}

					
					for(int k=0;k<m_itl[merageIndx1].rect_data.size();k++)
					{
						if(gap == 0 && k==0)
							continue;
						m_itl[merageIndx0].rect_data.push_back(m_itl[merageIndx1].rect_data[k]);
					}

					erase_map.push_back(m_itl[merageIndx1].id);					
				}
			}
			//while(i < N_MerageMap)
			//{
			//	if(m_itl[i].t_end >)
			//	{
			//		m_itl.erase(m_itl.begin() + i);
			//		N_MerageMap--;
			//		continue;
			//	}
			//	i++;
			//}
		}
	}

	if(erase_map.size()>0)
	{
		for(int k=0;k<erase_map.size();k++)
		{
			int N_m_itlh =m_itl.size();
			for(int i=0;i<N_m_itlh;i++)
			{
				if(m_itl[i].id == erase_map[k])
				{
					m_itl.erase(m_itl.begin()+i);
					N_m_itlh --;
				}
			}
		}
	}

	for(int i=0;i<m_MerageMap.size();i++ )
	{
		for(int k=0;k<m_detect_rect_squence.size();k++)
		{
			for(int j=0;j< m_detect_rect_squence[k].object_id.size();j++)
			{
				if(m_detect_rect_squence[k].object_id[j] == m_MerageMap[i][1])
				{
					m_detect_rect_squence[k].object_id[j] = m_MerageMap[i][0];
				}
			}
			
		}
	}


	//cout<<"m_process_frameNumCount"<<m_process_frameNumCount<<endl;

	//if(m_itl.size() >= 1)
	//{
	//	for(int i = 0; i < m_itl.size(); i++)
	//	{
	//		if(m_itl[i].length >= 3)
	//		{
	//			if(m_itl[i].t_end == m_process_frameNumCount)
	//			{
	//				Point x1;
	//				int idx=m_itl[i].xy_data.cols-1;
	//				x1.x = (int)m_itl[i].xy_data.at<float>(0, idx)*2;
	//				x1.y = (int)m_itl[i].xy_data.at<float>(1, idx)*2;
	//				cv::circle( m_frame, x1, 1, cv::Scalar( i * 20 & 255 , i * 20 & 255 , 0 ), 3 );
	//				char obj_id[255] = "0";
	//				sprintf_s(obj_id, "%d", m_itl[i].id);
	//				cv::putText(m_frame, obj_id, x1, FONT_HERSHEY_COMPLEX, 1, cv::Scalar( i * 20 & 255 , i * 20 & 255 , 0 ));
	//			}
	//			//imshow("m_frame", m_frame);
	//		}

	//	}

	//}

	//if(temp_c == ' ')
	//{
	//	temp_c = waitKey();
	//}
	//else
	//{
	//	temp_c = waitKey(1);
	//}
#if TRACK_DEBUG
	printf("track end\n");
#endif

    m_process_frameNumCount++;
	
    return 1;
}

//void CTracker::GetItlInfo(vector<ITL_BASE_INFO>&Track_Itl)
//{
//	DWORD id;
//	int t_start;
//	int t_end;
//	int length;
//	Mat omega;
//	Mat xy_data;
//	vector<Rect> rect_data;
//	Mat rect_id;
//	double rank;
//	int flag;
//
//	for (int i =0 ;i<m_itl.size();i++)
//	{
//		int  nLast = m_itl[i].rect_data.size()-1;
//		Rect RcLast= m_itl[i].rect_data[nLast];
//
//		ITL_BASE_INFO _itl;
//		_itl.id		  = m_itl[i].id;
//		_itl.t_start  = m_itl[i].t_start;
//		_itl.t_end	  = m_itl[i].t_end;
//		_itl.length	  = m_itl[i].length;
//		_itl.omega	  = m_itl[i].omega;
//		_itl.xy_data  = m_itl[i].xy_data;
//		_itl.rect_id  = m_itl[i].rect_id;
//		_itl.rank	  = m_itl[i].rank;
//		_itl.flag	  = m_itl[i].flag;
//		_itl.rect_Last=RcLast;
//		Track_Itl.push_back(_itl);
//	}
//}

int CTracker::InputDetectRect(vector<DETECTRECT> &_detect_rect_squence, int _frame_num, vector<Rect> _detect_rect)
{
#if TRACK_DEBUG
	printf("InputDetectRect\n");
#endif
    //队列不满
    if(_frame_num < m_nHistoryLen)
    {
        _detect_rect_squence[_frame_num].detect_rect = _detect_rect;
        for (int i = 0; i < _detect_rect.size(); i++)
        {
            //生成rect的IDX
            _detect_rect_squence[_frame_num].idx.push_back(i);
            Point center;
            center.x = _detect_rect_squence[_frame_num].detect_rect[i].x + _detect_rect_squence[_frame_num].detect_rect[i].width / 2;
            center.y = _detect_rect_squence[_frame_num].detect_rect[i].y + _detect_rect_squence[_frame_num].detect_rect[i].height / 2;
            _detect_rect_squence[_frame_num].detect_rect_center.push_back(center);
            _detect_rect_squence[_frame_num].object_id.push_back(0);
        }
        _detect_rect_squence[_frame_num].frame_num = _frame_num;
    }
    //队列满，擦处队列第一个
    else
    {
        _detect_rect_squence.erase(_detect_rect_squence.begin());
        DETECTRECT tempDetectRect;
        tempDetectRect.detect_rect = _detect_rect;
        for (int i = 0; i < _detect_rect.size(); i++)
        {
            //生成rect的IDX
            tempDetectRect.idx.push_back(i);
            Point center;
            center.x = tempDetectRect.detect_rect[i].x + tempDetectRect.detect_rect[i].width / 2;
            center.y = tempDetectRect.detect_rect[i].y + tempDetectRect.detect_rect[i].height / 2;
            tempDetectRect.detect_rect_center.push_back(center);
            tempDetectRect.object_id.push_back(0);
        }
        tempDetectRect.frame_num = _frame_num;

        _detect_rect_squence.push_back(tempDetectRect);
    }
    return 1;
}	

void  CTracker::DiffMat(Mat _a, Mat &_b) //求向量B的一阶差分 功能等价matlab里的diff
{
#if TRACK_DEBUG
	printf("DiffMat\n");
#endif
    int cols = _a.cols;
    int rows = _a.rows;
    int nChannels = _a.channels();
    if(rows != 1)
        _a = _a.reshape(_a.channels(), 1);
    _a.convertTo(_a, CV_32F);
    if(_b.empty())
        _b = Mat(1, cols - 1, CV_32F);
    float *pB = (float *)(_a.data);
    float *pOut = (float *)(_b.data);
    for(int i = 0; i < cols - 1; i++)
    {
        *pOut = *(pB + 1) - *pB;
        pB++;
        pOut++;
    }
#if TRACK_DEBUG
	printf("end-DiffMat\n");
#endif
}

void CTracker::NONUnique(Mat _a, Mat _distance, Mat &_b)
{
#if TRACK_DEBUG
	printf("NONUnique\n");
#endif
	if(m_process_frameNumCount ==24)
	{
		int klkl=0;
	}
    if(_a.empty())
        return;
    int rows = _a.rows;
    int cols = _a.cols;

    int rowvec = (rows == 1) && (cols > 1);
    //矩阵元素总数
    int numelA = cols * rows;
    if(numelA == 1)
    {
        _a.copyTo(_b);
        _b.convertTo(_b, CV_32F);
        return;
    }
    ////////cout<<_b<<endl;
    Mat tempA;
    tempA = _a.reshape(1, 1);
    Mat sorted;
    Mat sortedIdx;
    tempA.convertTo(tempA, CV_32F);
    cv::sort(tempA, sorted, CV_SORT_ASCENDING);
    cv::sortIdx(tempA, sortedIdx, CV_SORT_ASCENDING);
    //printf("DiffMat\n");
    Mat db = Mat();
    DiffMat(sorted, db);
    Mat d = Mat(1, cols * rows, CV_8U);
    Mat tempd = (db != 0);
    //cout<<"tempd="<<tempd<<endl;
    int tempd_size = tempd.channels() * tempd.cols * tempd.step * tempd.elemSize();
    tempd.copyTo(d.colRange(0, d.cols - 1));
    d.at<uchar>(0, cols * rows - 1) = 255;


    int nNonZero = countNonZero(d);
    int nZero = d.cols * d.rows - nNonZero;
    Mat tempB = Mat(1, nNonZero, CV_32F);

    int k = 0;
    for(int i = 0; i < rows * cols; i++)
    {
        if(d.at<uchar>(0, i) == 255)
        {
            if(k < nNonZero)
            {
                tempB.at<float>(0, k) = sorted.at<float>(0, i);
                k++;
            }
        }
    }

    //printf("nhistCount\n");
    //计算矩阵中各值的直方图
    vector<int> nhistCount(nNonZero);

    for(int k = 0; k < numelA; k++)
    {
        for(int i = 0; i < nNonZero; i++)
        {
            if(tempB.at<float>(0, i) == tempA.at<float>(0, k))
                nhistCount[i]++;
        }
    }
    //printf("重复元素进行压栈,%d\n",nNonZero);
    //重复元素进行压栈
    vector<float> tempVectorB;
    for(int k = 0; k < nNonZero; k++)
    {
        if(nhistCount[k] > 1)
        {
            tempVectorB.push_back(tempB.at<float>(0, k));
        }
    }

    if(tempVectorB.size() != 0)
    {
        //////cout<<"_distance="<<_distance<<endl;
        //////cout<<"tempA="<<tempA<<endl;
        //printf("重复元素的处理\n");
        for(int k = 0; k < tempVectorB.size(); k++)
        {
            if(tempVectorB[k] >= 0)
            {
                int min_indx = -1;
                double temp_distance = -1;
                //printf("重复元素的处理%d,tempVectorB.size=%d\n",k,tempVectorB.size());
                for(int j = 0; j < numelA; j++)
                {
                    //重复元素的处理
                    if(tempA.at<float>(0, j) == tempVectorB[k] )
                    {

                        //tempA.at<float>(0,j) = tempVectorB[k];
                        //距离远的置为-1，距离近的置为下一序
                        if(temp_distance == -1)
                        {
                            temp_distance = _distance.at<float>(j, tempVectorB[k]);
                            min_indx = j;
                        }
                        else
                        {
                            if(temp_distance > _distance.at<float>(j, tempVectorB[k]) )
                            {
                                temp_distance = _distance.at<float>(j, tempVectorB[k]);
                                min_indx = j;
                            }
                        }
                    }
                }

                for(int j = 0; j < numelA; j++)
                {
                    //重复元素的处理
                    if(tempA.at<float>(0, j) == tempVectorB[k] )
                    {
                        if(j != min_indx)
                        {
                            tempA.at<float>(0, j) = -1;
                        }
                    }
                }
            }
        }
    }
    tempA.copyTo(_b);
#if TRACK_DEBUG
	printf("end-NONUnique\n");
#endif
    ////////cout<<"tempA="<<tempA<<endl;
    /*}*/
}

int CTracker::CalRectDistance(DETECTRECT _detect_rect_t, DETECTRECT _detect_rect_tp1, int _ratio_threhold, Mat &_mat_distance, Mat &_rlt)
{
#if TRACK_DEBUG
	printf("CalRectDistance\n");
#endif
    int N_t = _detect_rect_t.detect_rect.size();
    //t+1时刻的检测框数量
    int N_tp1 = _detect_rect_tp1.detect_rect.size();
    if( N_t > 0 && N_tp1 > 0)
    {
        Mat mat_distance = Mat(N_t, N_tp1, CV_32FC1);
        //计算检测框间的距离
        for(int NT_i = 0 ; NT_i < N_t ; NT_i++)
        {
            for(int NT_ip1 = 0; NT_ip1 < N_tp1; NT_ip1++)
            {
                mat_distance.at<float>(NT_i, NT_ip1) = \
                pow((double)(_detect_rect_t.detect_rect_center[NT_i].x - _detect_rect_tp1.detect_rect_center[NT_ip1].x), 2)\
				+pow((double)(_detect_rect_t.detect_rect_center[NT_i].y - _detect_rect_tp1.detect_rect_center[NT_ip1].y), 2);
                mat_distance.at<float>(NT_i, NT_ip1) = sqrt(mat_distance.at<float>(NT_i, NT_ip1));
            }
        }
        mat_distance.copyTo(_mat_distance);

        ////////cout<<"_mat_distance="<<_mat_distance<<endl;
        //检测前两位距离比值
        Mat Dsorted;
        cv::sort(mat_distance, Dsorted, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING );
        //////cout<<"Dsorted="<<Dsorted<<endl;
        Mat DIdx;
        cv::sortIdx(mat_distance, DIdx, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING );
        //////cout<<"DIdx="<<DIdx<<endl;

        if(N_tp1 > 1)
        {
            Mat A = Dsorted.colRange(0, 1);
            Mat B = Dsorted.colRange(1, 2);
            Mat C;
            divide(B, A, m_Ratio);
            m_f = m_Ratio > _ratio_threhold;
            for(int i = 0; i < A.rows; i++)
            {
                if(A.at<float>(i, 0) == 0)
                    m_f.at<uchar>(i, 0) = 255;
            }
            minMaxLoc(Dsorted, NULL, &m_lastminD2, NULL, NULL); // 不需要的置为0
        }
        else if(m_lastminD2 > 0)
        {
            Mat A = Dsorted.colRange(0, 1);
            divide(m_lastminD2, A, m_Ratio);
            m_f = m_Ratio > _ratio_threhold;
            for(int i = 0; i < A.rows; i++)
            {
                if(A.at<float>(i, 0) == 0)
                    m_f.at<uchar>(i, 0) = 255;
            }
        }
        else
        {
            Mat A = Dsorted.colRange(0, 1);
            m_Ratio = A * 0;
            m_f = m_Ratio > _ratio_threhold;
        }

        Mat A = DIdx.colRange(0, 1);
        A.convertTo(A, CV_8UC1);

        Mat tempmat = A.mul(m_f / 255);
        Mat rlt;
        //////cout<<"DIdx="<<tempmat<<endl;
        ////////cout<<"_mat_distance"<<_mat_distance<<endl;
        //printf("NONUnique\n");
        //检测重复的idx
        NONUnique(tempmat, _mat_distance, rlt);

        //////cout<<"rlt="<<rlt<<endl;
        rlt.copyTo(_rlt);
    }
    else
    {
        return -1;
    }
    return 1;
}

int CTracker::FindAssociations(vector<DETECTRECT> &_detect_rect_squence, int _ratio_threhold, vector<Mat> &_b, vector<Mat> &_distance)
{
#if TRACK_DEBUG
	printf("FindAssociations\n");
#endif
    //序列不满足m_nHistoryLen长度的
    if(m_process_frameNumCount >= m_nHistoryLen )
    {
        _b.erase(_b.begin());
        _distance.erase(_distance.begin());
    }

    if(m_process_frameNumCount > 0)
    {
        int T = _b.size();
        //上一时刻的检测框数量
        int N_t = _detect_rect_squence[T].detect_rect.size();
        //当前时刻的检测框数量
        int N_tp1 = _detect_rect_squence[T - 1].detect_rect.size();
        if( N_t > 0 && N_tp1 > 0)
        {
            Mat mat_distance = Mat(N_t, N_tp1, CV_32FC1);
            Mat rlt;
            //printf("CalRectDistance\n");
            CalRectDistance(_detect_rect_squence[T], _detect_rect_squence[T - 1], _ratio_threhold, mat_distance, rlt);
            //CalRectDistance(_detect_rect_squence[T],_detect_rect_squence[T+1],_ratio_threhold,mat_distance,rlt);
            _distance.push_back(mat_distance);
            _b.push_back(rlt);

            //if(m_frame_num >=91)
            //{
            //	int ll=0;
            //	//////cout<<"_distance["<<T<<"]="<<_distance[T]<<endl;
            //	//////cout<<"_b["<<T<<"]="<<_b[T]<<endl;
            //}
            /*
            for(int i=0;i<_detect_rect_squence[T].detect_rect.size();i++)
            {
            rectangle(m_frame,_detect_rect_squence[T].detect_rect[i],Scalar(255,0,0));
            }

            for(int i=0;i<rlt.cols;i++)
            {
            int indx= rlt.at<float>(i);
            rectangle(m_frame,_detect_rect_squence[T+1].detect_rect[indx],Scalar(0,255,0));
            }
            ////imshow("m_frame",m_frame);
            waitKey(1);*/
        }
        else
        {
            N_tp1 = _detect_rect_squence[T].detect_rect.size();
            Mat tempmat = Mat(1, N_tp1, CV_32FC1);
            tempmat.setTo(-1);
            _b.push_back(tempmat);

            Mat temp_distance = Mat(1, 0, CV_32F);
            _distance.push_back(temp_distance);
        }
        if(N_t > N_tp1)
        {
            Mat tempmat = Mat(1, N_t, CV_32FC1);

        }
    }
    else if(m_process_frameNumCount == 0)
    {
        Mat tempmat = Mat(1, 0, CV_32F);
        _b.push_back(tempmat);

        Mat temp_distance = Mat(1, 0, CV_32F);
        _distance.push_back(temp_distance);
    }
    else
    {
        int T = _b.size() - 1;
        Mat tempmat = Mat(1, 0, CV_32F);
        tempmat.copyTo(_b[T]);

        Mat temp_distance = Mat(1, 0, CV_32F);
        temp_distance.copyTo(_distance[T]);
    }
    //else if(m_frame_num >= 2)
    //{
    //	//求取倒数第二帧的距离矩阵
    //	int T=m_nHistoryLen-2;
    //	int N_t=_detect_rect_squence[T].detect_rect.size();
    //	//t+1时刻的检测框数量
    //	int N_tp1=_detect_rect_squence[T+1].detect_rect.size();
    //	if( N_t > 0 && N_tp1> 0)
    //	{
    //		Mat mat_distance = Mat(N_t,N_tp1,CV_32FC1);
    //		Mat rlt;
    //		CalRectDistance(_detect_rect_squence[T],_detect_rect_squence[T+1],_ratio_threhold,mat_distance,rlt);

    //		//copy到对应的序列中
    //		mat_distance.copyTo(_distance[T-1]);
    //		rlt.copyTo(_b[T-1]);

    //		//////cout<<"_distance["<<T<<"]="<<_distance[T]<<endl;
    //		//////cout<<"_b["<<T<<"]="<<_b[T]<<endl;
    //		for(int i=0;i<_detect_rect_squence[T].detect_rect.size();i++)
    //		{
    //			rectangle(m_frame,_detect_rect_squence[T].detect_rect[i],Scalar(255,0,0));
    //		}

    //		for(int i=0;i<rlt.cols;i++)
    //		{
    //			int indx= rlt.at<float>(i);
    //			rectangle(m_frame,_detect_rect_squence[T+1].detect_rect[indx],Scalar(0,255,0));
    //		}
    //		////imshow("m_frame",m_frame);

    //	}
    //	//序列进行前移
    //	_distance.erase(_distance.begin());
    //	Mat temp_distance=Mat();
    //	_distance.push_back(temp_distance);
    //	_b.erase(_b.begin());
    //	N_tp1 = _detect_rect_squence[m_nHistoryLen-1].detect_rect.size();
    //	Mat tempmat=Mat(1,N_tp1,CV_32FC1);
    //	tempmat.setTo(-1);
    //	_b.push_back(tempmat);
    //}

    return 1;
}

//分段程序
//int CTracker::FindAssociations(vector<DETECTRECT> &_detect_rect_squence,int _ratio_threhold,vector<Mat> &_b,vector<Mat> &_distance)
//{
//	//序列不满足m_nHistoryLen长度的
//	if(m_frame_num >= m_nHistoryLen )
//	{
//		_b.erase(_b.begin());
//		_distance.erase(_distance.begin());
//	}
//
//
//	if(m_frame_num >= 1)
//	{
//		int T=_b.size()-1;
//		//上一时刻的检测框数量
//		int N_t=_detect_rect_squence[T].detect_rect.size();
//		//当前时刻的检测框数量
//		int N_tp1=_detect_rect_squence[T+1].detect_rect.size();
//		if( N_t > 0 && N_tp1> 0)
//		{
//			Mat mat_distance = Mat(N_t,N_tp1,CV_32FC1);
//			Mat rlt;
//			CalRectDistance(_detect_rect_squence[T+1],_detect_rect_squence[T],_ratio_threhold,mat_distance,rlt);
//			//CalRectDistance(_detect_rect_squence[T],_detect_rect_squence[T+1],_ratio_threhold,mat_distance,rlt);
//			mat_distance.copyTo(_distance[T]);
//			rlt.copyTo(_b[T]);
//	/*		//////cout<<"_distance["<<T<<"]="<<_distance[T]<<endl;
//			//////cout<<"_b["<<T<<"]="<<_b[T]<<endl;*/
//			/*
//			for(int i=0;i<_detect_rect_squence[T].detect_rect.size();i++)
//			{
//			rectangle(m_frame,_detect_rect_squence[T].detect_rect[i],Scalar(255,0,0));
//			}
//
//			for(int i=0;i<rlt.cols;i++)
//			{
//			int indx= rlt.at<float>(i);
//			rectangle(m_frame,_detect_rect_squence[T+1].detect_rect[indx],Scalar(0,255,0));
//			}
//			////imshow("m_frame",m_frame);
//			waitKey(1);*/
//		}
//
//		N_tp1 = _detect_rect_squence[T+1].detect_rect.size();
//		Mat tempmat=Mat(1,N_tp1,CV_32FC1);
//		tempmat.setTo(-1);
//		_b.push_back(tempmat);
//
//		Mat temp_distance=Mat(1,0,CV_32F);
//		_distance.push_back(temp_distance);
//	}
//	else if(m_frame_num == 0)
//	{
//		Mat tempmat=Mat(1,0,CV_32F);
//		_b.push_back(tempmat);
//
//		Mat temp_distance=Mat(1,0,CV_32F);
//		_distance.push_back(temp_distance);
//	}
//	else
//	{
//		int T=_b.size()-1;
//		Mat tempmat=Mat(1,0,CV_32F);
//		tempmat.copyTo(_b[T]);
//
//		Mat temp_distance=Mat(1,0,CV_32F);
//		temp_distance.copyTo(_distance[T]);
//	}
//	//else if(m_frame_num >= 2)
//	//{
//	//	//求取倒数第二帧的距离矩阵
//	//	int T=m_nHistoryLen-2;
//	//	int N_t=_detect_rect_squence[T].detect_rect.size();
//	//	//t+1时刻的检测框数量
//	//	int N_tp1=_detect_rect_squence[T+1].detect_rect.size();
//	//	if( N_t > 0 && N_tp1> 0)
//	//	{
//	//		Mat mat_distance = Mat(N_t,N_tp1,CV_32FC1);
//	//		Mat rlt;
//	//		CalRectDistance(_detect_rect_squence[T],_detect_rect_squence[T+1],_ratio_threhold,mat_distance,rlt);
//
//	//		//copy到对应的序列中
//	//		mat_distance.copyTo(_distance[T-1]);
//	//		rlt.copyTo(_b[T-1]);
//
//	//		//////cout<<"_distance["<<T<<"]="<<_distance[T]<<endl;
//	//		//////cout<<"_b["<<T<<"]="<<_b[T]<<endl;
//	//		for(int i=0;i<_detect_rect_squence[T].detect_rect.size();i++)
//	//		{
//	//			rectangle(m_frame,_detect_rect_squence[T].detect_rect[i],Scalar(255,0,0));
//	//		}
//
//	//		for(int i=0;i<rlt.cols;i++)
//	//		{
//	//			int indx= rlt.at<float>(i);
//	//			rectangle(m_frame,_detect_rect_squence[T+1].detect_rect[indx],Scalar(0,255,0));
//	//		}
//	//		////imshow("m_frame",m_frame);
//
//	//	}
//	//	//序列进行前移
//	//	_distance.erase(_distance.begin());
//	//	Mat temp_distance=Mat();
//	//	_distance.push_back(temp_distance);
//	//	_b.erase(_b.begin());
//	//	N_tp1 = _detect_rect_squence[m_nHistoryLen-1].detect_rect.size();
//	//	Mat tempmat=Mat(1,N_tp1,CV_32FC1);
//	//	tempmat.setTo(-1);
//	//	_b.push_back(tempmat);
//	//}
//
//	return 1;
//}

int CTracker::GetXYChain(vector<DETECTRECT> &_detect_rect_squence, vector<Mat> &_b, int _frame, int _ind, Mat &_xy)
{
#if TRACK_DEBUG
	printf("GetXYChain\n");
#endif
    int length = _b.size();
    int tfirst = _frame;
    Mat xy = Mat(7, length, CV_32FC1);
    xy.setTo(0);
    while(_frame < length - 1 && _ind >= 0)
    {
        Mat tempXYt = xy.colRange(_frame, _frame + 1);
        tempXYt.at<float>(0, 0) = (float)_detect_rect_squence[_frame].detect_rect_center[_ind].x;
        tempXYt.at<float>(1, 0) = (float)_detect_rect_squence[_frame].detect_rect_center[_ind].y;

        tempXYt.at<float>(2, 0) = (float)_detect_rect_squence[_frame].detect_rect[_ind].x;
        tempXYt.at<float>(3, 0) = (float)_detect_rect_squence[_frame].detect_rect[_ind].x;
        tempXYt.at<float>(4, 0) = (float)_detect_rect_squence[_frame].detect_rect[_ind].width;
        tempXYt.at<float>(5, 0) = (float)_detect_rect_squence[_frame].detect_rect[_ind].height;
        tempXYt.at<float>(6, 0) = (float)_detect_rect_squence[_frame].idx[_ind];

        if(_ind >= _b[_frame].cols)
            _ind = 0;
        int indnext = (int) _b[_frame].at<float>(0, _ind);
        _b[_frame].at<float>(0, _ind) = -1;
        _ind = indnext;

        _frame++;
    }
    _xy = xy.colRange(tfirst, _frame);
    return 1;
}

int CTracker::LinkDetectionTrackletsList(vector<DETECTRECT> &_detect_rect_squence, vector<Mat> _b, vector<Mat> _distance, vector<I_TRACK_LINK> &_itl)
{
#if TRACK_DEBUG
	printf("LinkDetectionTracklets\n");
#endif
    //////cout<<"*********************linkDetectionTracklets*******************"<<endl;
    _itl.clear();
    int length = _detect_rect_squence.size();
    int n = 1;
    int cols;
    int rows;
    Mat xy = Mat();
    vector<Mat> tempvector;

    for (int i = 0; i < _b.size(); i++)
    {
        Mat temp_mat;
        _b[i].copyTo(temp_mat);
        tempvector.push_back(temp_mat);
        //if(m_frame_num<95 && m_frame_num>71)
        //	//////cout<<temp_mat<<",";
    }

    for(int i = 0; i < _b.size(); i++)
    {
        if(tempvector[i].cols == 0 || tempvector[i].rows == 0)
            continue;
        cols = tempvector[i].cols;
        for(int k = 0; k < cols; k++)
        {
            if(tempvector[i].at<float>(0, k) > -1)
            {
                GetXYChain(_detect_rect_squence, tempvector, i, k, xy);
                I_TRACK_LINK tempITL;
                int l = xy.cols;

                tempITL.t_start = i + _detect_rect_squence[0].frame_num;
                tempITL.t_end = tempITL.t_start + l - 1;
                tempITL.length = l;
                tempITL.omega = Mat(1, l, CV_32FC1);
                tempITL.omega.setTo(1);
                //xy:前两行中心点坐标，后四行为rect区域
                Mat temp_xy_data = xy.rowRange(0, 2);
                Mat temp_rect_data = xy.rowRange(2, 6);
                temp_xy_data.copyTo(tempITL.xy_data);
                //			temp_rect_data.copyTo(tempITL.rect_data);
                _itl.push_back(tempITL);
            }
        }
    }
    //if(m_frame_num >=92 && m_frame_num <= 96)
    //{
    //	//////cout<<"**********xy_data**************"<<endl;
    //	for(int i=0;i<_itl.size();i++)
    //	{
    //		//////cout<<_itl[i].xy_data<<endl;
    //	}
    //	//////cout<<"**********xy_data**************"<<endl;
    //}

    tempvector.clear();
    return 1;
}

int CTracker::LinkDetectionTracklets(vector<DETECTRECT> &_detect_rect_squence, vector<Mat> _b, vector<Mat> _distance, vector<I_TRACK_LINK> &_itl)
{
	int b_length = _b.size();
	int T = b_length - 1;
	int Tnp1 = T - 1;
	if(b_length < 2)
		return -1;

	//当前T时刻,b_length-2
	int T_b_cols = _b[T].cols;
	int T_b_rows = _b[T].rows;

	//T-1时刻,b_length-1
	int Tnp1_b_cols = _b[Tnp1].cols;
	int Tnp1_b_rows = _b[Tnp1].rows;
	//没有跟踪目标
	if(T_b_cols == 0 || T_b_rows == 0)
		return -1;
#if TRACK_DEBUG
//	printf("目标关联\n");
#endif

	//发现跟踪目标，查找与T-1时刻的关系
	//T-1为空，新发现目标
	for(int i = 0; i < T_b_cols; i++)
	{
		int object_id = 0;
		int ind = _b[T].at<float>(0, i);
#if TRACK_DEBUG
	//	printf("T-1为空，新发现目标\n");
#endif

		
		//T-1为空，新发现目标
		if(ind == -1)
		{
			object_id = GetTickCount();
#if TRACK_DEBUG
			object_id = m_process_frameNumCount+ i;
#endif
			_detect_rect_squence[T].object_id[i] = object_id;
			Mat tempXYt = Mat(2, 1, CV_32FC1);
			tempXYt.at<float>(0, 0) = (float)_detect_rect_squence[T].detect_rect_center[i].x;
			tempXYt.at<float>(1, 0) = (float)_detect_rect_squence[T].detect_rect_center[i].y;

			Rect data_rect;
			data_rect.x = (float)_detect_rect_squence[T].detect_rect[i].x;
			data_rect.y = (float)_detect_rect_squence[T].detect_rect[i].y;
			data_rect.width = (float)_detect_rect_squence[T].detect_rect[i].width;
			data_rect.height = (float)_detect_rect_squence[T].detect_rect[i].height;

			I_TRACK_LINK new_itl;
			new_itl.id = object_id;
			new_itl.t_start = _detect_rect_squence[T].frame_num;
			new_itl.t_end = new_itl.t_start;
			new_itl.length = 1;
			new_itl.omega = Mat(1,1,CV_8U);
			new_itl.omega.setTo(1);
			tempXYt.copyTo(new_itl.xy_data);
			new_itl.rect_data.push_back(data_rect);
			if(_itl.size() ==0)
				new_itl.flag = -1;

			_itl.push_back(new_itl);

			//_b[T].at<float>(0, i) = -2;
		}
		else
		{
			object_id = _detect_rect_squence[Tnp1].object_id[ind];
			_detect_rect_squence[T].object_id[i] = _detect_rect_squence[Tnp1].object_id[ind];
			for(int j = 0; j < _itl.size(); j++)
			{
				if(object_id == _itl[j].id)
				{
					_itl[j].length ++;
					_itl[j].t_end = _detect_rect_squence[T].frame_num;

					Mat tempXYt = Mat(2, 1, CV_32FC1);
					tempXYt.at<float>(0, 0) = (float)_detect_rect_squence[T].detect_rect_center[i].x;
					tempXYt.at<float>(1, 0) = (float)_detect_rect_squence[T].detect_rect_center[i].y;

					Rect data_rect;
					data_rect.x = (float)_detect_rect_squence[T].detect_rect[i].x;
					data_rect.y = (float)_detect_rect_squence[T].detect_rect[i].y;
					data_rect.width = (float)_detect_rect_squence[T].detect_rect[i].width;
					data_rect.height = (float)_detect_rect_squence[T].detect_rect[i].height;

					Mat temp_meger = Mat(2, _itl[j].length, CV_32FC1);
					_itl[j].xy_data.copyTo(temp_meger.colRange(0, _itl[j].length - 1));
					tempXYt.copyTo(temp_meger.colRange(_itl[j].length - 1, _itl[j].length));
					temp_meger.copyTo(_itl[j].xy_data);

					Mat tempOmega= Mat(1, 1, CV_8U);
					tempOmega.setTo(1);
					Mat temp_omega_meger = Mat(1, _itl[j].length, CV_8U);
					_itl[j].omega.copyTo(temp_omega_meger.colRange(0, _itl[j].length - 1));
					tempOmega.copyTo(temp_omega_meger.colRange(_itl[j].length - 1, _itl[j].length));
					temp_omega_meger.copyTo(_itl[j].omega);

					_itl[j].rect_data.push_back(data_rect);

				}
			}
		}
	}
	return 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
int growitl(vector<I_TRACK_LINK> _itl, int max_D)
{
    int N_itl = _itl.size();
    BOOL loop_done = FALSE;

    int gap = 0;
    while (!loop_done)
    {
        for(int i = 0; i < N_itl; i++)
        {
            for(int j = 0; j < N_itl; j++)
            {
                gap = (_itl[j].t_start - _itl[i].t_end) - 1;
                if(gap == 0)
                {
                    if(_itl[i].length == 1 && _itl[j].length > 1)
                    {

                    }
                }
            }
        }
    }

    return 1;
}

int CTracker::Get_ITL_Horizon(vector<I_TRACK_LINK> _itl, int _t_start, int _t_end, vector<I_TRACK_LINK> &_itlh)
{
#if TRACK_DEBUG
	printf("Get_ITL_Horizon\n");
#endif
    int N = _itl.size();

    int hormin = _t_start;
    int hormax = _t_end;
    vector<BOOL> f(N);

    for(int i = 0; i < N; i++)
    {
        f[i] = !(_itl[i].t_start >= hormax || _itl[i].t_end <= hormin);
        if(f[i] == 1)
        {
            _itlh.push_back(_itl[i]);
            _itlh[i].id = i;
        }
    }

    for(int i = 0; i < _itlh.size(); i++)
    {
        int si = max(hormin - _itlh[i].t_start, 0);
        int ei = max(_itlh[i].t_end - hormax, 0);

        int cols = _itlh[i].xy_data.cols;
        Mat tempData = _itlh[i].xy_data.colRange(si, cols - ei);
        tempData.copyTo(_itlh[i].xy_data);
        ////////cout<<_itlh[i].xy_data<<endl;
        cols = _itlh[i].omega.cols;
        Mat tempOmega = _itlh[i].omega.colRange(si, cols - ei);
        tempOmega.copyTo(_itlh[i].omega);
        ////////cout<<_itlh[i].omega<<endl;
        _itlh[i].t_start = max(_itlh[i].t_start, hormin);
        _itlh[i].t_end = min(_itlh[i].t_end , hormax);

        _itlh[i].length = _itlh[i].t_end - _itlh[i].t_start + 1;
    }

    return 1;
}

int CTracker::circshift(Mat &_s, int _nr = 0, int _nc = 0)
{
    //旋转矩阵
    //NR按行滚动nr次，正数自上往下滚动，负数自下往上滚动
    //NR按列滚动nr次，正数自左往右滚动，负数自右往左滚动
    Mat temp_mat = Mat();
    _s.copyTo(temp_mat);

    int rows = _s.rows;
    int cols = _s.cols;
    int row_times = _nr % rows;
    int col_times = _nc % cols;

    //处理行的滚动
    if(row_times > 0)
    {
        Mat temp_s_mat1 = _s.rowRange(0, rows - row_times);
        Mat temp_s_mat2 = _s.rowRange(rows - row_times, rows);

        Mat temp_temp_mat1 = temp_mat.rowRange(0, row_times);
        Mat temp_temp_mat2 = temp_mat.rowRange(row_times, rows);

        temp_s_mat1.copyTo(temp_temp_mat2);
        temp_s_mat2.copyTo(temp_temp_mat1);


    }
    else if(row_times < 0)
    {
        int temp_row_times = -row_times;
        Mat temp_s_mat1 = _s.rowRange(0, temp_row_times);
        Mat temp_s_mat2 = _s.rowRange(temp_row_times, rows);

        Mat temp_temp_mat1 = temp_mat.rowRange(rows - temp_row_times, rows);
        Mat temp_temp_mat2 = temp_mat.rowRange(0, rows - temp_row_times);

        temp_s_mat1.copyTo(temp_temp_mat1);
        temp_s_mat2.copyTo(temp_temp_mat2);
    }

    //处理行的滚动
    if(col_times > 0)
    {
        Mat temp_s_mat1 = _s.colRange(0, cols - col_times);
        Mat temp_s_mat2 = _s.colRange(cols - col_times, cols);

        Mat temp_temp_mat1 = temp_mat.colRange(0, col_times);
        Mat temp_temp_mat2 = temp_mat.colRange(col_times, cols);

        temp_s_mat1.copyTo(temp_temp_mat2);
        temp_s_mat2.copyTo(temp_temp_mat1);


    }
    else if(col_times < 0)
    {
        int temp_col_times = -col_times;
        Mat temp_s_mat1 = _s.colRange(0, temp_col_times);
        Mat temp_s_mat2 = _s.colRange(temp_col_times, cols);

        Mat temp_temp_mat1 = temp_mat.colRange(cols - temp_col_times, cols);
        Mat temp_temp_mat2 = temp_mat.colRange(0, cols - temp_col_times);

        temp_s_mat1.copyTo(temp_temp_mat1);
        temp_s_mat2.copyTo(temp_temp_mat2);
    }

    temp_mat.copyTo(_s);
    return 1;

}

int CTracker::form_S(Mat &_S, int _nr, int _nc, int _D)
{
#if TRACK_DEBUG
	printf("form_S\n");
#endif
    int N = _nr / _D + _nc - 1;
    int dims = 2;
    int size[] = {_nr * _nc, N * _D};
    SparseMat S_sparse(dims, size, CV_8U);
    Mat S_full;
    S_sparse.convertTo(S_full, CV_8U);

    Mat s = Mat(Size(N * _D, _nr), CV_8U);
    s.setTo(0);

    Mat temp_s = s.colRange(0, _nr);
    temp_s = Mat::eye(_nr, _nr, CV_8U);

    for(int i = 0; i < _nc; i++)
    {
        Mat temp_S = S_full.rowRange(i * _nr, (i + 1) * _nr);
        s.copyTo(temp_S);
        circshift(temp_S, 0, i * _D);
    }

    S_full.convertTo(_S, CV_8U);
#if TRACK_DEBUG
	printf("end--form_S\n");
#endif
    return 1;
}

int CTracker::form_P(Mat &_P, Mat _omega, int _D, int _N)
{
#if TRACK_DEBUG
	printf("form_P\n");
#endif
    int dims = 2;
    int size[] = {_D * _N, _D * _N};
    SparseMat P_sparse(dims, size, CV_32F);
    //P(1:N*D+1:N^2*D^2) = reshape(repmat(Omega,[D 1]),[1 N*D]);

    Mat temp_P;
    P_sparse.convertTo(temp_P, CV_32F);
    int element_count = temp_P.total();

    Mat temp_mat;
    repeat(_omega, _D, 1, temp_mat);
    temp_mat = temp_mat.t();
	temp_mat.convertTo(temp_mat,CV_32F);
    Mat temp_mat1 = temp_mat.reshape(temp_mat.channels(), 1);
	
    int indx_k = 0;
    for(int i = 0; indx_k < temp_mat1.total(); i = i + _N * _D + 1)
    {
        temp_P.at<float>(i) = temp_mat1.at<float>(indx_k);
        indx_k++;
    }
    temp_P.copyTo(_P);

#if TRACK_DEBUG
//	cout<<temp_mat1<<endl;
	printf("end--form_P\n");
#endif
    return 1;
}

int CTracker::fastshrink(Mat _R, double _th, Mat &_J)
{
    Mat R ;
    _R.convertTo(R, CV_32F);
    SVD svd_R;
    Mat U, Vt, temp_S;
    svd_R.compute(R, temp_S, U, Vt, SVD::FULL_UV );

    Mat S = Mat(U.rows, Vt.cols, CV_32F);
    S.setTo(0);
    for(int i = 0; i < temp_S.total(); i++)
    {
        S.at<float>(i, i) = temp_S.at<float>(i);
    }
    int M = S.rows;
    int N = S.cols;

    for(int i = 0; i < M * N; i = i + N + 1)
    {
        S.at<float>(i) = S.at<float>(i) - _th;
    }
    max(S, 0, S);
    Mat A = U * S * Vt;
    A.copyTo(_J);
    return 1;
}

int CTracker::L2_fastalm_mo(I_TRACK_LINK _itl, RESULTS _p)
{
    int D = _itl.xy_data.rows;
    int N = _itl.xy_data.cols;
    int nr = ceil((double)N / (D + 1)) * D;
    int nc = N - ceil((double)N / (D + 1)) + 1;
    int defMaxIter = 50;
    double defTol = 0.0000001;

    Mat Omega  = Mat();
    int MaxIter = defMaxIter;
    double Tol = defTol;

    if(_p.nr == INF)
        _p.nr = nr;
    else
        nr =  _p.nr;

    if(_p.nc == INF)
        _p.nc = nc;
    else
        nc =  _p.nc;

    if(_p.nc == INF)
        _p.nc = nc;
    else
        nc =  _p.nc;

    if(_itl.omega.empty())
        Omega = Mat::ones(1, N, CV_8U);
    else
        _itl.omega.copyTo(Omega);

    Size JSize = Size(nr, nc);

    if(_p.MaxIter != defMaxIter)
        MaxIter = defMaxIter;

    if(_p.Tol != defTol)
        Tol = _p.Tol;
    Mat vec_u = _itl.xy_data.t();
    vec_u = vec_u.reshape(1, 1);
    vec_u.convertTo(vec_u, CV_32F);

    Mat S;
    form_S(S, nr, nc, D);
    S.convertTo(S, CV_32F);
    Mat P;
    form_P(P, Omega, D, N);
    P.convertTo(P, CV_32F);

    Mat PtP = P.t() * P;
    Mat StS = S.t() * S;

    Mat diag_PtP = PtP.diag();
    Mat diag_StS = StS.diag();

    double max_vec_u = 0;
    minMaxLoc(vec_u, NULL, &max_vec_u, NULL, NULL);
    double mu = 0.05 / (max_vec_u / 10);
    double rho = 1.05;
    Mat h = 1.1 * vec_u;
    h = h.t();
    Mat y = Mat(Size(1, nr * nc), CV_32F);
    y = y.setTo(0);
    Mat R = Mat(JSize, CV_32F);
    R = R.t();
    R = R.setTo(0);
    int R_rows = R.rows;

    for(int i = 0; i < MaxIter; i++)
    {
        Mat temp_mat1 = S * h;
        Mat temp_mat2 ;
        divide(y, mu, temp_mat2);

        add(temp_mat1, temp_mat2, R);
        R = R.reshape(1, R_rows);
        R = R.t();

        Mat J;
        fastshrink(R, 1 / mu, J);

        Mat j = J.t();
        j = j.reshape(1, 1);
        j = j.t();

        Mat temp_h1 = mu * S.t() * (j - y / mu);
        temp_h1 = temp_h1 + (_p.lambda * PtP * vec_u.t()) ;
        Mat temp_h2 = (_p.lambda * diag_PtP + mu * diag_StS);
        divide(temp_h1, temp_h2, h);

        Mat temp_mat = S * h - j;
        y = y + mu * temp_mat;
        mu = mu * rho;

        if(norm(temp_mat) < Tol)
            break;
    }
    Mat  u_hat = h.reshape(1, N);
    u_hat = u_hat.t();
    u_hat.copyTo(_itl.xy_data);
    return 1;
}

int  CTracker::hankel_mo(Mat _itl_xy, int _nr, int  _nc, Mat &_D, Mat &_H)
{
    int dim = _itl_xy.rows;
    int N = _itl_xy.cols;

    if(_nr == 0)
        _nr = (N - _nc + 1) * dim;

    if(_nc == 0)
    {
        _nr = _nc;
        _nc = N - _nr / dim + 1;

        if((int)_nr % dim != 0)
            printf("error\n");
    }

    int nb = (int)_nr / dim;
    int l = MIN(nb, (int)_nc);
    int D_length = l - 1 + N - 2 * l + 2 + l - 1;

    _D = Mat(1, D_length, CV_8U);
    _D.setTo(l);
    for(int i = 0; i < l - 1; i++)
    {
		_D.at<uchar>(i)=i+1;
		_D.at<uchar>(l - 1 + N - 2 * l + 2 + i) =l - 1 - i;
    }
    Mat cidx = Mat(1, _nc, CV_8U);
    for(int i = 0; i < cidx.cols; i++)
    {
        cidx.at<uchar>(i) = i;
    }
    Mat tempCidx = cv::repeat(cidx, _nr, 1);
    ////////cout<<"tempCidx"<<tempCidx<<endl;
    Mat ridx = Mat(1, _nr , CV_8U);
    for(int i = 0; i < ridx.cols; i++)
    {
        ridx.at<uchar>(i) = i + 1;
    }
    ridx = ridx.t();
    Mat tempRidx = cv::repeat(ridx, 1, _nc);
    addWeighted(tempRidx, 1, tempCidx, dim, 0, _H);
    _H = _H - 1;
	_H.convertTo(_H,CV_32F);
    //注意变换顺序，MATLAB和OPENCV的转换不同
    Mat tempT = Mat();
    tempT = _itl_xy.t();
    tempT = tempT.reshape(0, 1);
    ////////cout<<tempT<<endl;

    Mat tempSubs = Mat();
    tempSubs = _H.t();
    tempSubs = tempSubs.reshape(1, 1);
    //tempSubs = tempSubs -1;
    tempSubs.convertTo(tempSubs, CV_32F);
    _H = Mat::zeros(_nc, _nr, CV_32F);
    for(int i = 0; i < _nr * _nc; i++)
    {
        int idx = tempSubs.at<float>(i);
        _H.at<float>(i) = tempT.at<float>(idx);
    }
    _H = _H.t();
    return 1;
}

int CTracker::smot_rank_admm(I_TRACK_LINK _itl, int _eta, RESULTS _p)
{
#if TRACK_DEBUG
	printf("smot_rank_admm start\n");
#endif
    int D = _itl.xy_data.rows;
    int N = _itl.xy_data.cols;

    int nr = ceil((double)N / (D + 1)) * D;
    int nc = N - ceil((double)N / (D + 1)) + 1;

    int defMaxRank = MIN(nr, nc);
    int defMinRank = 1;
    double defLambda  = 0.1;

    int R_max = 0;
    if(_p.max_rank == INF)
        R_max = defMaxRank;
    else
        R_max = _p.max_rank;
    R_max = MIN(R_max, nr);
    R_max = MIN(R_max, nc);

    int R_min = 0;
    if(_p.min_rank == INF)
        R_min = defMinRank;
    else
        R_min = _p.min_rank;

    double Lambda = 0;
    if(_p.lambda != defLambda)
        Lambda = _p.lambda;
    else
        Lambda = defLambda;

    Mat omega = Mat();
    if (_itl.omega.empty())
    {
        omega = Mat::ones(1, N, CV_8U);
    }
    else
    {
        _itl.omega.copyTo(omega);
    }

    int nCount_total_omega = omega.total();
    int nCount_zero_omega = 0;
    for(int i = 0; i < nCount_total_omega; i++)
    {
        if(omega.at<uchar>(i) == 0)
            nCount_zero_omega++;
    }

    if( nCount_zero_omega > 0)
        L2_fastalm_mo(_itl, _p);
    Mat matH = Mat();
    Mat matD = Mat();
    hankel_mo(_itl.xy_data, 0, R_max, matD, matH);

    SVD matH_SVD(matH);
    int nCount_matH_SVDW = matH_SVD.w.total();
    int total_matH_SVDW = 0;
    for(int i = 0; i < nCount_matH_SVDW; i++)
    {
        if(matH_SVD.w.at<float>(i) > _eta)
        {
            total_matH_SVDW ++;
        }

    }
    double R = MAX(R_min, total_matH_SVDW);
    R = MIN(R_max, R);
#if TRACK_DEBUG
	printf("smot_rank_admm end\n");
#endif
    return (int)R;
}

int CTracker::smot_similarity(I_TRACK_LINK _itl_xy1, I_TRACK_LINK _itl_xy2, int _eta, int _gap, RESULTS &_p1, RESULTS &_p2, double &_rank12, double &_s)
{

#if TRACK_DEBUG
	printf("smot_similarity start\n");
#endif
    int D1 = _itl_xy1.xy_data.rows;
    int T1 = _itl_xy1.xy_data.cols;

    int D2 = _itl_xy2.xy_data.rows;
    int T2 = _itl_xy2.xy_data.cols;

    if(D1 != D2)
        printf("Error:Input dimensions do not agree.\n");

    double defGap      = 0;
    Mat Omega1   = Mat();
    Mat Omega2   = Mat();
    double defRank1    = INF;
    double defRank2    = INF;
    BOOL defQCheck   = false;

    double gap = 0;
    if(_gap == 0)
        gap = defGap;
    else
        gap = _gap;

    if(_itl_xy1.omega.empty())
        Omega1   = Mat::ones(1, T1, CV_8U);
    else
        _itl_xy1.omega.copyTo(Omega1);

    if(_itl_xy2.omega.empty())
        Omega2   = Mat::ones(1, T1, CV_8U);
    else
        _itl_xy2.omega.copyTo(Omega2);

    double rank1 = 0;
    if(_p1.rank == INF)
        rank1 = defRank1;
    else
        rank1 = _p1.rank;

    double rank2 = 0;
    if(_p2.rank == INF)
        rank2 = defRank2;
    else
        rank2 = _p2.rank;
    //修改全局变量
    BOOL qcheck = _p1.qcheck;

    _p1.rank = (double)smot_rank_admm(_itl_xy1, _eta, _p1);
    _p2.rank = (double)smot_rank_admm(_itl_xy2, _eta, _p2);

    rank1 = _p1.rank ;
    rank2 = _p2.rank ;
    if(qcheck)
    {
#if TRACK_DEBUG
		printf("qcheck\n");
#endif
        int nr = (int)MIN(T1 - rank1, T2 - rank2);
        nr = nr / D1 * D1;
        Mat H1 = Mat();
        Mat H2 = Mat();
        Mat tempD;
        hankel_mo(_itl_xy1.xy_data, nr, 0, tempD, H1);
        hankel_mo(_itl_xy1.xy_data, nr, 0, tempD, H2);
        Mat combineMat;
        combineMat.push_back(H1);
        combineMat.push_back(H2);
        SVD combineMat_svd(combineMat);
        int nCount = combineMat_svd.w.total();
        double  sum = 0;
        for(int i = 0; i < nCount; i++)
        {
            if(combineMat_svd.w.data[i] > _eta)
                sum = sum  + combineMat_svd.w.data[i];
        }
        _rank12 = sum;

        if(_rank12 > rank1 + rank2)
        {
            _s = -INF;
            return 1;
        }
#if TRACK_DEBUG
		printf("end-qcheck\n");
#endif
    }

    Mat tempMat = Mat();

    Mat XY12_data = Mat();
    tempMat = _itl_xy1.xy_data.t();
    XY12_data.push_back(tempMat);
    tempMat = Mat::zeros((int)gap, (int)D1, CV_32F);
    XY12_data.push_back(tempMat);
	tempMat = _itl_xy2.xy_data.t();
    XY12_data.push_back(tempMat);

    XY12_data = XY12_data.t();
	

    Mat Omega12 = Mat();
    tempMat = Omega1.t();
    Omega12.push_back(tempMat);
    tempMat = Mat::zeros((int)gap, 1, CV_8U);
    Omega12.push_back(tempMat);
    tempMat = Omega2.t();
    Omega12.push_back(tempMat);
    Omega12 = Omega12.t();

    I_TRACK_LINK itl_xy12;
    XY12_data.copyTo(itl_xy12.xy_data);
    Omega12.copyTo(itl_xy12.omega);
    RESULTS p;
    p.min_rank = MIN(rank1, rank2);
    p.max_rank = rank1 + rank2;


    _rank12 = smot_rank_admm(itl_xy12, _eta, p);
#if TRACK_DEBUG
	printf("end-_rank12-smot_similarity\n");
#endif
	// _s = (rank1 + rank2) / _rank12 - 1;
    _s = (rank1 + rank2) / _rank12 ;
    if(_s < 0.000005)
        _s = -INF;
#if TRACK_DEBUG
	printf("smot_similarity-end\n");
#endif
    //binlong ：to be continue
    return 1;
}

void CTracker::similarity_itl(I_TRACK_LINK &_itl_i, I_TRACK_LINK &_itl_j, DEFAULT_PARAMS _params, double &_rank12, double &_s)
{
#if TRACK_DEBUG
	printf("similarity_itl start\n");
#endif
    double defMaxHorizon = INF;
    double defMaxGap = INF;
    double defMaxSlope = INF;

    double hor_max = 0;
    if(_params.hor_max == INF)
        hor_max = defMaxHorizon;
    else
        hor_max = _params.hor_max;

    double gap_max = 0;
    if(_params.gap_max == INF)
        gap_max = defMaxGap;
    else
        gap_max = _params.gap_max;

    double slope_max = 0;
    if(_params.slope_max == INF)
        slope_max = defMaxSlope;
    else
        slope_max = _params.slope_max;

    // Individual Ranks
    double rank1 = _itl_i.rank;
    double rank2 = _itl_j.rank;
    double rank12 = INF;

	if(_itl_i.length <= 2 && _itl_j.length <= 2)
		return ;

	//_params.gap = _itl_j.t_start - _itl_i.t_end - 1;
    _params.gap = _itl_j.t_start - _itl_i.t_end ;
    double slope = 0;
    Mat tempMat = Mat();
    addWeighted(_itl_j.xy_data.col(0), 1, _itl_i.xy_data.col(_itl_i.xy_data.cols - 1), -1, 0, tempMat);
    slope = norm(tempMat, 4) / (_params.gap + 1);


    RESULTS _itl_p1;
    RESULTS _itl_p2;
    if( _params.gap >= 0 && _params.gap < _params.gap_max && slope <= (_params.slope_max * 2))
        smot_similarity(_itl_i, _itl_j, _params.eta_max, _params.gap, _itl_p1, _itl_p2, _rank12, _s);
    else if (0 & (_itl_j.t_start > _itl_i.t_start ) && (_itl_j.t_end < _itl_i.t_end))
    {
        int i = 0;
        int rt_start = _itl_j.t_start - _itl_i.t_start + 1;
        int rt_end = rt_start + _itl_j.length - 1;
        int total_sum = 0;
        for(int i = 0; i < rt_end - rt_start; i++)
        {
            if(_itl_i.omega.at<float>(i) == 0)
                total_sum++;
        }
        int rmax = 0;
        if(total_sum == _itl_j.length)
            rmax = rank1 + rank2;
		smot_similarity(_itl_i, _itl_j, _params.eta_max, _params.gap, _itl_p1, _itl_p2, _rank12, _s);
        /*

        if sum(itl1.omega(rt_start:rt_end)==0) == itl2.length
        % compute the joint rank
        rmax = r1+r2;

        % TODO: Rewrite the following
        [s r1 r2 r12] = smot_similarity([itl1.data(:,1:rt_start-1) itl2.data itl1.data(:,rt_end+1:end)],eta_max,...
        'gap',gap,...
        'rank1',itl1.rank,'rank2',itl2.rank,...
        'omega1',itl1.omega,'omega2',itl2.omega);

        %'omega',[itl1.omega(1:rt_start-1) itl2.omega itl1.omega(rt_end+1:end)]);


        end*/
    }

#if TRACK_DEBUG
	printf("end--similarity_itl\n");
#endif
}

int CTracker::Compute_ITL_Similarity_Matrix(vector<I_TRACK_LINK> &_itl, DEFAULT_PARAMS _param)
{
    int N = _itl.size();

    double max_gap = 0;
    int nCount = 0;
    for(int i = 0; i < N; i++)
    {
        if(_itl[i].length > 1)
        {
            max_gap = max_gap + _itl[i].length;
            nCount++;
        }
    }

    max_gap = max_gap / nCount / 2;

    double max_slope = 0;
    //////cout<<"********************************************"<<endl;
    int cols = 0;
    for(int i = 0; i < N; i++)
    {
        cols = _itl[i].xy_data.cols;
        Mat tempData1 = _itl[i].xy_data.colRange(0, cols - 1);
        Mat tempData2 = _itl[i].xy_data.colRange(1, cols);

        Mat dx;
        cv::absdiff(tempData1, tempData2, dx);
        Mat norm_dx = dx.mul(dx);
        norm_dx.convertTo(norm_dx, CV_32F);
        Mat sum = Mat::zeros(1, norm_dx.cols, CV_32F);
        for (int i = 0; i < norm_dx.rows; i++)
        {
            add(norm_dx.row(i), sum, sum);
        }

        sqrt(sum, norm_dx);
        //////cout<<norm_dx<<endl;
        Mat maxMat;
        max(norm_dx, max_slope, maxMat);
        //////cout<<"maxMat"<<maxMat<<endl;
        minMaxLoc(maxMat, NULL, &max_slope, NULL, NULL);
    }

    Mat matS = Mat::ones(N, N, CV_32F);
    matS.setTo(NINF);


    _param.slope_max = max_slope;
    _param.gap_max = max_gap;

    double rank12 = INF;
    double s = 0;
    for(int i = 0; i < N; i = i++)
    {
        for(int j = 0; j < N; j++)
        {
            if( i == j)
                s = NINF;
            else
            {
                similarity_itl(_itl[i], _itl[j], _param, rank12, s);
            }
            matS.at<float>(i, j) = (float)s;
        }
    }
	
//	cout<<matS<<endl;
    return 1;
}


int CTracker::Compute_ITL_Param(vector<I_TRACK_LINK> _itl,DEFAULT_PARAMS& _param)
{
	int N = _itl.size();

	double max_gap = 0;
	int nCount = 0;
	for(int i = 0; i < N; i++)
	{
		if(_itl[i].length > 1)
		{
			max_gap = max_gap + _itl[i].length;
			nCount++;
		}
	}

	max_gap = max_gap / nCount / 2;

	double max_slope = 0;
	//////cout<<"********************************************"<<endl;
	int cols = 0;
	for(int i = 0; i < N; i++)
	{
		cols = _itl[i].xy_data.cols;
		Mat tempData1 = _itl[i].xy_data.colRange(0, cols - 1);
		Mat tempData2 = _itl[i].xy_data.colRange(1, cols);
		/*cout<<_itl[i].xy_data<<endl;
		cout<<tempData1<<endl;
		cout<<tempData2<<endl;*/
		Mat dx;
		cv::absdiff(tempData1, tempData2, dx);
		Mat norm_dx = dx.mul(dx);
		norm_dx.convertTo(norm_dx, CV_32F);
		Mat sum = Mat::zeros(1, norm_dx.cols, CV_32F);
		for (int i = 0; i < norm_dx.rows; i++)
		{
			add(norm_dx.row(i), sum, sum);
		}

		sqrt(sum, norm_dx);
		//cout<<norm_dx<<endl;
		Mat maxMat;
		max(norm_dx, max_slope, maxMat);
		//////cout<<"maxMat"<<maxMat<<endl;
		minMaxLoc(maxMat, NULL, &max_slope, NULL, NULL);
	}

	_param.slope_max = max_slope;
	_param.gap_max = max_gap;

	return 1;
}

int  CTracker::Associate_ITL(vector<I_TRACK_LINK> _itl)
{
    int N = _itl.size();
    //	vector<I_TRACK_LINK> itlh;

    //去除过短的跟踪线
    int N_itlh = _itl.size();
	int i = 0;
	while(i < N_itlh)
	{
 		if(_itl[i].length < 4)
		{
			_itl.erase(_itl.begin() + i);
			N_itlh--;
			continue;
		}
		i++;
	}
	
	for(int k=0;k<N_itlh;k++)
	{
		if(_itl[k].length > m_nHistoryLen*2)
		{
			Mat temp_xy_data = _itl[k].xy_data.colRange(_itl[k].length-m_nHistoryLen*2,_itl[k].length);
			temp_xy_data.copyTo(_itl[k].xy_data);
			Mat temp_omega   = _itl[k].omega.colRange(_itl[k].length-m_nHistoryLen*2,_itl[k].length);
			temp_omega.copyTo(_itl[k].omega);

			_itl[k].length  = m_nHistoryLen*2;
			_itl[k].t_start = _itl[k].t_end - m_nHistoryLen*2+1;
			

		}
	}

	if(N_itlh > 0)
	{

		DEFAULT_PARAMS params;
		Compute_ITL_Param(_itl,params);
		m_MerageMap.clear();

		for (i=N_itlh-1;i>=0;i--)
		{
			
			if(_itl[i].flag == 1 )
			{
				double rank12 = INF;
				double s = 0;
				for(int k =0;k<i;k++)
				{
					similarity_itl(_itl[k], _itl[i], params, rank12, s);
					if(s != 0 &&  s != NINF)
					{
						_itl[i].flag = -1;
						vector<DWORD> temp_map(2);
						temp_map[0] = _itl[k].id;
						temp_map[1] = _itl[i].id;
						m_MerageMap.push_back(temp_map);
						break;
					}
				}

				if(_itl[i].flag == 1)
				{
					_itl[i].flag = -1;
				}
			}
		}
	}

	

    //int Nnew = 0;
    //int dN = 1;
    //if(!_itl.empty())
    //{
    //    while(dN > 0)
    //    {
    //        Compute_ITL_Similarity_Matrix(_itl, params);
    //        Nnew = _itl.size();
    //        dN = N - Nnew;
    //        N = Nnew;
    //    }
    //}
    return 1;
}

int CTracker::Compute_DetectionTracklets_Similarity(vector<I_TRACK_LINK> &_itl)
{
    //无可跟踪目标
    if(m_itl.size() <= 0 && _itl.size() <= 0)
    {
        m_itl.clear();
		return 1;
    }

	

    return 1;
}


