#For the use of cross slope analysis
from cmath import nan
import os,glob,sys
from turtle import forward
import numpy as np
import pandas
from scipy.io import loadmat
from matplotlib import pyplot as plt
from sklearn import linear_model
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
import pyproj
import shapefile as pyshp
import copy
import pandas as pd
ZONE = 11
pj = pyproj.Proj(proj='utm', zone=ZONE, ellps='WGS84')
OUTPUT_FOLDER = r'Y:\Users\Trevor\crossSlope'
LINE_FOLDER = r'Y:\Users\Trevor\crossSlope'
SPACING = 20 #Meter sapcing of segments
LONGITUDE_SEARCH = SPACING/2 #Meter sapcing of segments
FINDPMARKS = False
DOVISUALIZE = False

sys.path.append(r'y:\reza\bin\bin\repos\core_dev_kit\LAS_kit')
# import LAS_CLASS 
sys.path.append(r'y:\reza\bin\bin\repos\core_dev_kit\TIFF_kit')
import TIFF_CLASS 
lr = linear_model.LinearRegression()
ransac = linear_model.RANSACRegressor()

def findChainage(xx,yy):
    # assuming that the xx,yy are sorted along the rbarr
    dx=np.diff(xx)
    dy=np.diff(yy)
    ch=np.cumsum(np.sqrt(dx**2+dy**2))
    ch=np.insert(ch,0,0)
    return ch

def resample_lat_lon(lat,lon,resampling_spacing):
    pj = pyproj.Proj(proj='utm', zone=ZONE, ellps='WGS84')
    lato = []
    lono = []
    xx,yy = pj(lon,lat)
    ch = findChainage(xx,yy)
    chFun = interp1d(ch,[xx,yy],kind='linear')
    ran = False
    for chi in np.arange(ch.min(),ch.max(),resampling_spacing):# every 1m along the rbarr
        loci = chFun(chi)
        xxi = loci[0]
        yyi = loci[1]
        loni,lati = pj(xxi,yyi,inverse=True)
        lono.append(loni)
        lato.append(lati)
        ran = True
    if ran:# make sure that last point is included
        if chi<ch.max():
            chi = ch.max()
            loci = chFun(chi)
            xxi = loci[0]
            yyi = loci[1]
            loni,lati = pj(xxi,yyi,inverse=True)
            lono.append(loni)
            lato.append(lati)
    else:
        lato = lat
        lono = lon
    lato = np.array(lato)
    lono = np.array(lono)
    return lato,lono,ch

def findLSRSLPT(lpt):
    ls_lat = []
    ls_lon = []
    rs_lat = []
    rs_lon = []
    ls_off = []
    rs_off = []
    lpt = pyshp.Reader(lpt)
    Shapes = lpt.shapes()
    Records = lpt.records()
    Fields = lpt.fields[1:]
    Fields = [Fields[0] for Fields in Fields] 
    xpsindex = Fields.index('XPS')
    offsetindex = Fields.index('DISTANCE2I')
    OKlist = ['Left', 'Right', 'left', 'right']
    for line in range(len(Records)):
        if Records[line][xpsindex].lower() in OKlist:
            lonlat  = np.array(Shapes[line].points)
            lon  = lonlat[:,0]
            lat  = lonlat[:,1]
            lat,lon,chainage = resample_lat_lon(lat,lon,0.5)

            offset = [Records[line][offsetindex]] * len(lon) # right now offset is value of line to icc, not used
            # lat,lon,chainage = resample_lat_lon(lat,lon,0.5) #FIX could take out depeding added
            ee,nn = pj(lon,lat)
            if 'ight' in Records[line][xpsindex].lower():
                rs_lat= np.concatenate((rs_lat, ee))
                rs_lon= np.concatenate((rs_lon, nn)) 
                rs_off = np.concatenate((rs_off, offset))     
            if 'eft' in Records[line][xpsindex].lower():
                ls_lat= np.concatenate((ls_lat, ee))
                ls_lon= np.concatenate((ls_lon, nn))
                ls_off = np.concatenate((ls_off, offset)) 

    lsds = np.vstack([np.array(ls_lat),np.array(ls_lon)]).T
    treeLS = KDTree(lsds)
    rsds = np.vstack([np.array(rs_lat),np.array(rs_lon)]).T
    treeRS  = KDTree(rsds)
    return treeRS,treeLS,rs_off,ls_off,np.array(rs_lat),np.array(rs_lon),np.array(ls_lat),np.array(ls_lon)

def loadICC(path2icc):  
    icc = loadmat(path2icc,struct_as_record=False,squeeze_me=True)
    return icc['icc_data']

def exact(las,id_valid_below_sen1,id_valid_below_sen2):
    slopedata = r"Y:\2021\TRN.PAVE03181-03_SEAHD\MDR\20210920_7001\21604p3a_xslope_01.mat"
    slopedata = loadmat(slopedata,struct_as_record=False,squeeze_me=True)
    xslope_data = slopedata['xslope_data']
    print('here')
    # for q in range(len(xslope_data.lat)):
    #     if not np.isnan(xslope_data.xs_15[q]):
    #         print('%s%s'%(xslope_data.xs_15[q]*100,'%'))
    #         index,distance = las.find_lat_lon(xslope_data.lat[q],xslope_data.lon[q])
    #         km = (las.lasin.tt_chainage_m[index])/500
    #         id_valid_km = las.filter_by_chainage(MIN=km-LONGITUDE_SEARCH,MAX=km+LONGITUDE_SEARCH)#Filter in the section we are looking at, multiplies by 500
    #         id_valid_below_sen1_KM = id_valid_below_sen1 & id_valid_km
    #         id_valid_below_sen2_KM = id_valid_below_sen2 & id_valid_km
    #         getCrossSlope(las,id_valid_below_sen1_KM,id_valid_below_sen2_KM,survey_name,km,1.25,2.75)
    df = pd.read_csv(r"Y:\2021\TRN.PAVE03181-03_SEAHD\MDR\20210920_7001\21604P3A_iri20.csv")
    dicti = {'kms':[],'Compare':[],'S1Fitted':[],'S1HalvedLane':[],'S2Fitted':[],'S2HalvedLane':[]}
    for q in range(len(df['Road_No'])):
        print(df['Crown_Slope'][q])
        if q == 0:
            continue
        elif q == 1:
            index,distance = las.find_lat_lon(df['From_Latitude'][q],df['From_Longitude'][q])
            km = ((las.lasin.tt_chainage_m[index])/500)[0]
            realkm = df['From_km_DMI'][q] *1000
            diff = km - realkm
        print(df['From_km_DMI'][q] *1000,km-diff)
        id_valid_km = las.filter_by_chainage(MIN=km-LONGITUDE_SEARCH,MAX=km+LONGITUDE_SEARCH)#Filter in the section we are looking at, multiplies by 500
        id_valid_below_sen1_KM = id_valid_below_sen1 & id_valid_km
        id_valid_below_sen2_KM = id_valid_below_sen2 & id_valid_km
        slopes = getCrossSlope(las,id_valid_below_sen1_KM,id_valid_below_sen2_KM,survey_name,km,1.25,2.75)
        km = km + 20
        dicti['kms'].append(df['From_km_DMI'][q])
        dicti['Compare'].append(df['Crown_Slope'][q])
        dicti['S1Fitted'].append(slopes[0])
        dicti['S1HalvedLane'].append(slopes[1])
        dicti['S2Fitted'].append(slopes[2])
        dicti['S2HalvedLane'].append(slopes[3])

    newdf = pd.DataFrame(dicti)
    newdf.to_csv(r'Y:\Users\Trevor\crossSlope\csvs\21604P3A_iri20.csv',index=False)

def consistency(las,id_valid_below_sen1,id_valid_below_sen2,dicti):
    dicti['kms'] = []
    dicti['Compare'] = []
    dicti['S1Fitted_%s'%survey_name] = []
    dicti['S1HalvedLane_%s'%survey_name] = []
    dicti['S2Fitted_%s'%survey_name] = []
    dicti['S2HalvedLane_%s'%survey_name] = []

    for q in range(0,550,50):
        if q == 0:
            # index,distance = las.find_lat_lon(54.145862,-112.475727)
            # index,distance = las.find_lat_lon(53.439763,-112.875465)
            index,distance = las.find_lat_lon(51.843821,-111.091651)

            km = ((las.lasin.tt_chainage_m[index])/500)[0]
            realkm = 0
            diff = km - realkm
        # print(km-diff)
        id_valid_km = las.filter_by_chainage(MIN=km-LONGITUDE_SEARCH,MAX=km+LONGITUDE_SEARCH)#Filter in the section we are looking at, multiplies by 500
        id_valid_below_sen1_KM = id_valid_below_sen1 & id_valid_km
        id_valid_below_sen2_KM = id_valid_below_sen2 & id_valid_km
        print(q)
        slopes = getCrossSlope(las,id_valid_below_sen1_KM,id_valid_below_sen2_KM,survey_name,q,1.75,2.25)
        km = km + SPACING
        dicti['kms'].append(q)
        dicti['Compare'].append(None)
        dicti['S1Fitted_%s'%survey_name].append(slopes[0])
        dicti['S1HalvedLane_%s'%survey_name].append(slopes[1])
        dicti['S2Fitted_%s'%survey_name].append(slopes[2])
        dicti['S2HalvedLane_%s'%survey_name].append(slopes[3])
    return dicti

def useBins(las,id_valid_below_sen1,id_valid_below_sen2,idbin):
    for km_num in range(idbin.min(),idbin.max()+1): #Itterate over each segment of road 
        print('\t BIN:%02d'%km_num)
        id_valid_km = idbin==km_num
        id_valid_below_sen1_KM = id_valid_below_sen1 & id_valid_km
        id_valid_below_sen2_KM = id_valid_below_sen2 & id_valid_km
        getCrossSlope(las,id_valid_below_sen1_KM,id_valid_below_sen2_KM,survey_name,km_num,1.25,2.75)

def useICC(las,id_valid_below_sen1,id_valid_below_sen2,dicti,path2icc):
    dicti['Route'] = []
    dicti['kms'] = []
    dicti['lat'] = []
    dicti['lon'] = []
    dicti['elv'] = []
    dicti['Super_Elevation'] = []
    dicti['Crown_Slope'] = []
    dicti['Compare'] = []
    dicti['S1Fitted_%s'%survey_name] = []
    dicti['S1HalvedLane_%s'%survey_name] = []
    dicti['S2Fitted_%s'%survey_name] = []
    dicti['S2HalvedLane_%s'%survey_name] = []
    
    # distrs = 1.25
    # distls = 2.50
    # distrs = 1.75
    # distls = 2.00
    # distrs = 1.5
    # distls = 1.5
    distrs = 2.0
    distls = 2.0
    if FINDPMARKS:
        # pmarkshp = r"Y:\AT_2021\ASSETS\pmarks\newMethod\20220204\merged\63002E_C1R1_R1R2.shp"
        pmarkshp = os.path.join(LINE_FOLDER,path2icc.replace('m','shp'))
        if os.path.exists(pmarkshp):
            treeRS,treeLS,rs_off,ls_off,rs_ee,rs_nn,ls_ee,ls_nn = findLSRSLPT(pmarkshp)
        else:
            print('NO PMARKS AVAILABLE %s'%pmarkshp)

    icc = loadICC(path2icc)
    # arcIcc = loadmat(path2icc.replace('icc','chg'),struct_as_record=False,squeeze_me=True)['arc_chg']
    # dmiIcc = loadmat(path2icc.replace('icc','chg'),struct_as_record=False,squeeze_me=True)['dmi_chg']
    dmichain = (icc.ppgps.dmi * icc.chg_data.dx)
    dmiFun = interp1d(dmichain,np.arange(0,len(dmichain)))
    lat = icc.ppgps.lat
    lon = icc.ppgps.lon
    # DMIee,DMInn = pj(lon,lat)
    # treedmi = KDTree(np.array([DMIee,DMInn]).T)

    startChainage =  icc.refrst * icc.chg_data.dx
    endChainage =  icc.secend * icc.chg_data.dx
    st=startChainage
    en=endChainage
    if startChainage>endChainage:
        st = endChainage
        en = startChainage
    stdmi = int(dmiFun(st))
    lat = icc.ppgps.lat[stdmi]
    lon = icc.ppgps.lon[stdmi]
    lasindex,lasdistance = las.find_lat_lon(lat,lon)#Point of 0m
    if lasdistance > 10:
        print('wrong point %s'%lasdistance)
        return
    laschain = ((las.lasin.tt_chainage_m[lasindex])/TT_SCALE)[0]
    if False: #Input what SHOULD be the start point in lat/lon, for testing
        # manuallatst = 51.843821 #63002
        # manuallonst = -111.091651
        manuallatst = 52.463126
        manuallonst = -113.774405
        manualindex,manualdistance = las.find_lat_lon(manuallatst,manuallonst)#Point of 0m
        manualchain = (las.lasin.tt_chainage_m[manualindex])/TT_SCALE
        print('manually checked start = %s    auto start = %s'%(manualchain[0],laschain))

    for xdmi  in np.arange(st,en,SPACING): 
        idmi = int(dmiFun(xdmi))
        # idmi2 = int(dmiFun(xdmi+SPACING))
        # idmi2 = min(idmi2,len(dmichain-2))
        # dmi = icc.ppgps.dmi[idmi]
        lat = icc.ppgps.lat[idmi]
        lon = icc.ppgps.lon[idmi]
        elv = icc.ppgps.elev[idmi]
        # if dmi>icc.refrst and dmi<icc.secend:
        if True:
            if FINDPMARKS:
                ee,nn = pj(lon,lat)
                distrs,rsindex = treeRS.query([ee,nn])
                distls,lsindex = treeLS.query([ee,nn])
                print('LS:%.3f RS:%.3f'%(distls,distrs))
                if distls>1.5 or distls<0.65:
                    if distls>1.5:
                        distls = 1.5
                    if distls<0.65:
                        distls = 0.65
                if distrs>3 or distrs<1.5:
                    if distrs>3:
                        distrs = 3
                    if distrs<1.5:
                        distrs = 1.5
                '''THERE IS A OFFSET DISCREPANCY BETWEEN THE LIDAR AND THE MDR LOCATION'''
                ofst = 1#Rough estimate, looks pretty close
                distrs += -ofst
                distls += ofst
                print('LS:%.3f RS:%.3f  CHANGEDTO'%(distls,distrs))
                # buff = 0.2 #Extra little buffer
                # distls += buff 
                # distrs += buff 

            id_valid_km = las.filter_by_chainage(MIN=laschain-LONGITUDE_SEARCH,MAX=laschain+LONGITUDE_SEARCH)#Filter in the section we are looking at, multiplies by 500
            # id_valid_km = las.filter_by_chainage(MIN=kmcheck2,MAX=kmcheck2+SPACING)#Filter in the section we are looking at, multiplies by 500
            # km = kmcheck2 - OG + 27300#For val site
            km = xdmi - st
            print('\nmdrkm = %s   laskm = %s'%(km,laschain))
            id_valid_below_sen1_KM = id_valid_below_sen1 & id_valid_km
            id_valid_below_sen2_KM = id_valid_below_sen2 & id_valid_km
            slopes, slopetype = getCrossSlope(las,id_valid_below_sen1_KM,id_valid_below_sen2_KM,survey_name,km,distls,distrs)
            dicti['Route'].append(path2icc)
            dicti['kms'].append(km/1000)
            dicti['lat'].append(lat)
            dicti['lon'].append(lon)
            dicti['elv'].append(lon)
            dicti['Compare'].append(None)
            dicti['S1Fitted_%s'%survey_name].append(slopes[0])
            dicti['S1HalvedLane_%s'%survey_name].append(slopes[1])
            dicti['S2Fitted_%s'%survey_name].append(slopes[2])
            dicti['S2HalvedLane_%s'%survey_name].append(slopes[3])
            if slopetype == 'Crown':
                dicti['Super_Elevation'].append(None)
                dicti['Crown_Slope'].append(np.mean(slopes))
            else:
                dicti['Super_Elevation'].append(np.mean(slopes))
                dicti['Crown_Slope'].append(None)
            if startChainage>endChainage:
                laschain = (laschain-SPACING)
            else:
                laschain = (laschain+SPACING)
    return dicti

def findType(slptypesample,trans_X,elev_Z,lsz,rsz):
    leftsidez = np.mean(slptypesample)
    if rsz > lsz or (rsz < lsz and leftsidez > lsz):#Use to compare to the 'crown' point, needs work
        slopetype = 'Super'
    else:
        slopetype = 'Crown'
        index_max = np.argmax(elev_Z)+1#Make sure the crown does not pass over to other side
        if index_max < len(elev_Z):
            elev_Z = elev_Z[:index_max]
            trans_X = trans_X[:index_max]
    return trans_X, elev_Z, slopetype

def method1(las,valid,distls,distrs):
    '''This method takes an average of the longitudial points ang plots on a transverse'''
    #Need something here to select points a certain distance away, get the height
    offsets = las.lasin.tt_trans_offset_m[valid]
    points = las.lasin.z[valid]
    
    x = list(np.arange(-6,8,0.1))#Not sure about side
    # x = list(np.arange(-distrs,distls,0.1))#Not sure about side
    z = []
    trans_X = []
    elev_Z = []
    slptypesample = []
    for xx in x:
        '''Goes back to Full, wastes time'''
        # id_valid = las.filter_by_offset(OFFSET=xx)
        # newValid = valid & id_valid
        # points = las.lasin.z[newValid]
        # y.append(np.mean(points))
        '''Just does slice'''
        # id_valid = offsets== xx*TT_SCALE
        # id_valid = abs(offsets - xx*TT_SCALE) < 1 #Not sure about this range yet...
        id_valid = abs(offsets/TT_SCALE - xx) < 0.05 #Not sure about this range yet...
        p = points[id_valid]
        if len(p) < 1:
            z.append(nan)
            continue
        avpoint = np.mean(p)
        z.append(avpoint)

        if xx > -distrs and xx < distls:
            trans_X.append(xx)
            elev_Z.append(avpoint)
            '''Get the end points'''
            lsz = avpoint
            if len(trans_X) == 1:
                rsz = avpoint
        if 3.25 < xx < 4.25:
            slptypesample.append(avpoint)


    #Split lane in half reading
    trans_X, elev_Z, slopetype = findType(slptypesample,trans_X,elev_Z,lsz,rsz)
    halfdist = (distrs+distls)/2
    half1 = np.mean(elev_Z[:len(trans_X)//2])
    half2 = np.mean(elev_Z[len(trans_X)//2:])
    newslope = ((half2-half1)/halfdist) *100

    model = lr.fit(np.array(trans_X).reshape((-1, 1)),np.array(elev_Z))
    slope = model.coef_[0] *100
    # print('Fit Line Reading = %3.3f%s'%(slope*100,'%'))
    # print('Half Lane Split  = %3.3f%s'%(newslope*100,'%'))
    # diff = ((maxpoints - minpoints)/((distls+distrs)*2))*100
    # print('Ends %s%s'%(diff,'%'))
    visualxy = [trans_X,elev_Z]

    return slope, newslope, visualxy, x, z, slopetype

def method2(las,valid,distls,distrs):
    '''This method splits into longitudinal sections, takes those slopes than does a final average at the end'''
    listslopes = []
    listnewslopes = []
    chainages = las.lasin.tt_chainage_m[valid]
    offsets = las.lasin.tt_trans_offset_m[valid]
    points = las.lasin.z[valid]   
    x = list(np.arange(-6,8,0.1))#Looking across the transverse
    y = list(np.arange(chainages.min(),chainages.max(),1*TT_SCALE)) 
    z = [] #for viewing
    slptypesample = []
    visualxy = [[],[]]
    for yy in y:
        middle = False
        if yy == chainages.min() + LONGITUDE_SEARCH*TT_SCALE:#Visualize just the middle cross ection to check
            middle= True
        trans_X = []
        elev_Z = []
        id_valid1 = (yy <= chainages)
        id_valid2 = (chainages <= yy+1*TT_SCALE)
        p = points[(id_valid1)&(id_valid2)]
        for xx in x:
            # id_valid3 = abs(offsets - xx*TT_SCALE) < 1 #Not sure about this range yet...
            id_valid3 = abs(offsets/TT_SCALE - xx) < 0.05 #Not sure about this range yet...
            p = points[(id_valid1)&(id_valid2)&(id_valid3)]
            if len(p) < 1:
                if middle:
                    z.append(nan)
                continue
            avpoint = np.mean(p)
            if xx >= -distrs and xx <= distls:
                trans_X.append(xx)
                elev_Z.append(avpoint)  
                lsz = avpoint
                if len(visualxy[0]) == 1:
                    rsz = avpoint
                if middle:
                    visualxy[0].append(xx)
                    visualxy[1].append(avpoint)  
            if 3.25 < xx < 4.25:
                slptypesample.append(avpoint)
            if middle:
                z.append(avpoint)

        trans_X, elev_Z, slopetype = findType(slptypesample,trans_X,elev_Z,lsz,rsz)
        halfdist = (distrs+distls)/2
        half1 = np.mean(elev_Z[:len(trans_X)//2])
        half2 = np.mean(elev_Z[len(trans_X)//2:])
        newslope = ((half2-half1)/halfdist)
        model = lr.fit(np.array(trans_X).reshape((-1, 1)),np.array(elev_Z))
        slope = model.coef_[0]
        listslopes.append(slope*100)
        listnewslopes.append(newslope*100)
    slope = np.mean(listslopes)
    newslope = np.mean(listnewslopes)

    return slope, newslope, visualxy, x, z, slopetype
    
def getCrossSlope(las,id_valid_below_sen1_KM,id_valid_below_sen2_KM,survey_name,km_num,distls,distrs):
    slopes = []
    # outname_sen1_KM = SURVERY_OUTPUT_FOLDER+r'\%s_KM%03d_sen01.las'%(survey_name,km_num)
    # outname_sen2_KM = SURVERY_OUTPUT_FOLDER+r'\%s_KM%03d_sen02.las'%(survey_name,km_num)
    outname_sen1_KM = SURVERY_OUTPUT_FOLDER+r'\%s_%05dm_sen01.las'%(survey_name,km_num)
    outname_sen2_KM = SURVERY_OUTPUT_FOLDER+r'\%s_%05dm_sen02.las'%(survey_name,km_num)

    outnames = [outname_sen1_KM,outname_sen2_KM]
    valids = [id_valid_below_sen1_KM,id_valid_below_sen2_KM]
    # outnames = [outname_sen1_KM]
    # valids = [id_valid_below_sen1_KM]
    for a in range(len(outnames)):#Itterate over each sensor
        outname = outnames[a]
        valid = valids[a]

        if np.sum(valid)>0:
            if not os.path.exists(outname) or True:
                
                slope, newslope, visualxy, x, z, slopetype = method1(las,valid,distls,distrs)
                # slope, newslope, visualxy, x, z, slopetype = method2(las,valid,distls,distrs)

                print('Sensor%d  %s Fit Line Reading = %3.3f%s   Half Lane Split  = %3.3f%s'%(a+1,slopetype,slope,'%',newslope,'%'))  
                slopes.append(slope)
                slopes.append(newslope)                  
                if DOVISUALIZE:#Do a visulaize output
                    visualize(outname,visualxy[0],visualxy[1],x,z,distls,distrs,slope,newslope)
                
            else:
                print('Already done: ',outname)
        else:
            print('Not Valid: ',outname)
            slopes.append(None)
            slopes.append(None)
    return slopes, slopetype

def visualize(outname,trans_X,long_Y,x,y,distls,distrs,slope,newslope):
    '''XY is points of slope calclation, xy are from all the points available'''
    halfidx = trans_X[len(trans_X)//2]
    halfdist = (distrs+distls)/2
    half1 = np.mean(long_Y[:len(trans_X)//2])
    half2 = np.mean(long_Y[len(trans_X)//2:])
    #Ransac visual
    ransac.fit(np.array(trans_X).reshape((-1, 1)),np.array(long_Y))
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    plt.scatter(np.array(trans_X).reshape((-1, 1))[inlier_mask], np.array(long_Y)[inlier_mask], color="yellowgreen", marker=".")
    plt.scatter(np.array(trans_X).reshape((-1, 1))[outlier_mask], np.array(long_Y)[outlier_mask], color="gold", marker=".")

    #Half lane averages
    plt.plot([-distrs,halfidx], [half1,half1], color="cyan")
    plt.plot([halfidx,distls], [half2,half2], color="pink")

    #Values taken boundary
    plt.plot([distls,distls], [max(y),min(y)], color="y")
    plt.plot([-distrs,-distrs], [max(y),min(y)], color="r")

    #Fitted line
    # line_X = np.arange(np.array(trans_X).reshape((-1, 1)).min(), np.array(trans_X).reshape((-1, 1)).max())[:, np.newaxis]
    # line_y = lr.predict(line_X)
    # plt.plot(line_X, line_y, color="orange")

    #Plot
    plt.title('%s\n %3.3f%s %3.3f%s'%(os.path.basename(outname),slope,'%',newslope,'%'))
    plt.plot(x, y, color="navy", label="just a line")
    plt.gca().invert_xaxis()
    plt.savefig(outname.replace('.las','.png'))
    # plt.show()
    plt.close()
    # print(outname)
    # las.export_to_las(las.lasin.points[valid],outname)

def extractCrossSlope(path2las,path2icc):
    dicti = {}
    survey_name = os.path.basename(path2las)[:-4]
    date_folder = os.path.basename(os.path.dirname(path2las))
    OUTPUT_FOLDER_DATED = os.path.join(OUTPUT_FOLDER,date_folder)
    if not os.path.exists(OUTPUT_FOLDER_DATED):
        os.mkdir(OUTPUT_FOLDER_DATED)
        global SURVERY_OUTPUT_FOLDER
    SURVERY_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_DATED,survey_name)
    print(SURVERY_OUTPUT_FOLDER)
    if not os.path.exists(SURVERY_OUTPUT_FOLDER):
        os.mkdir(SURVERY_OUTPUT_FOLDER)
    print('[Processing] >> ',path2las)

    las = LAS_CLASS.LASREAD(path2las)
    # filters here
    _,idbelow = las.split_top_bottom_las()#Filter below sensor
    idsen1 , idsen2 = las.split_sensors()#Split sensors due to mismatch
    # idvalid = las.filter_by_tt_trans(RANGE=offset)#Filter on either side of van
    idvalid = las.filter_by_tt_trans(RANGE=8)#Filter on either side of van
    idbin,bin_count = las.bin_by_chainage(BINSIZE=SPACING)# Every x meter create a chunk
    global TT_SCALE
    TT_SCALE = las.get_scale()#Get the scale factor

    id_valid_below_sen1 = (idbelow)&(idsen1)&(idvalid)
    id_valid_below_sen2 = (idbelow)&(idsen2)&(idvalid)
    # las.export_to_las(id_valid_below_sen1,os.path.join(SURVERY_OUTPUT_FOLDER,))#Something to save the LAS
    # las.export_to_las(id_valid_below_sen2,os.path.join(SURVERY_OUTPUT_FOLDER,))#Something to save the LAS


    # if True: 
    #     dicti = consistency(las,id_valid_below_sen1,id_valid_below_sen2,dicti)
    # exact(las,id_valid_below_sen1,id_valid_below_sen2)
    # if doBin:
    #     print('total bins',bin_count)
    #     useBins(las,id_valid_below_sen1,id_valid_below_sen2,idbin)
    dicti = useICC(las,id_valid_below_sen1,id_valid_below_sen2,dicti,path2icc)
    return dicti, survey_name

def useICCTIFF(path2TiffFolder,survey_name,dicti,path2icc):
    dicti['Route'] = []
    dicti['kms'] = []
    dicti['lat'] = []
    dicti['lon'] = []
    dicti['elv'] = []
    dicti['Super_Elevation'] = []
    dicti['Crown_Slope'] = []
    dicti['Compare'] = []
    dicti['S1Fitted_%s'%survey_name] = []
    dicti['S1HalvedLane_%s'%survey_name] = []
    dicti['S2Fitted_%s'%survey_name] = []
    dicti['S2HalvedLane_%s'%survey_name] = []

    # distrs = 1.25
    # distls = 2.50
    # distrs = 1.75
    # distls = 2.00
    # distrs = 1.5
    # distls = 1.5
    distrs = 2.0
    distls = 2.0
    if FINDPMARKS:
        # pmarkshp = r"Y:\AT_2021\ASSETS\pmarks\newMethod\20220204\merged\63002E_C1R1_R1R2.shp"
        pmarkshp = os.path.join(LINE_FOLDER,path2icc.replace('m','shp'))
        if os.path.exists(pmarkshp):
            treeRS,treeLS,rs_off,ls_off,rs_ee,rs_nn,ls_ee,ls_nn = findLSRSLPT(pmarkshp)
        else:
            print('NO PMARKS AVAILABLE %s'%pmarkshp)

    icc = loadICC(path2icc)
    # arcIcc = loadmat(path2icc.replace('icc','chg'),struct_as_record=False,squeeze_me=True)['arc_chg']
    # dmiIcc = loadmat(path2icc.replace('icc','chg'),struct_as_record=False,squeeze_me=True)['dmi_chg']
    dmichain = (icc.ppgps.dmi * icc.chg_data.dx)
    dmiFun = interp1d(dmichain,np.arange(0,len(dmichain)))
    lat = icc.ppgps.lat
    lon = icc.ppgps.lon
    # DMIee,DMInn = pj(lon,lat)
    # treedmi = KDTree(np.array([DMIee,DMInn]).T)

    startChainage =  icc.refrst * icc.chg_data.dx
    endChainage =  icc.secend * icc.chg_data.dx
    st=startChainage
    en=endChainage
    if startChainage>endChainage:
        st = endChainage
        en = startChainage
    stdmi = int(dmiFun(st))
    lat = icc.ppgps.lat[stdmi]
    lon = icc.ppgps.lon[stdmi]

    posit = False
    tiffList = sorted(glob.glob(os.path.join(path2TiffFolder,'*_sen00_*ZSTD*.tiff')), reverse=True)
    if posit:
        tiffList = sorted(glob.glob(os.path.join(path2TiffFolder,'*_sen00_*ZSTD*.tiff')))


    # lasindex,lasdistance = las.find_lat_lon(lat,lon)#Point of 0m
    # if lasdistance > 10:
    #     print('wrong point %s'%lasdistance)
    #     return
    # laschain = ((las.lasin.tt_chainage_m[lasindex])/TT_SCALE)[0]
    if False: #Input what SHOULD be the start point in lat/lon, for testing
        # manuallatst = 51.843821 #63002
        # manuallonst = -111.091651
        manuallatst = 52.463126
        manuallonst = -113.774405
        manualindex,manualdistance = las.find_lat_lon(manuallatst,manuallonst)#Point of 0m
        manualchain = (las.lasin.tt_chainage_m[manualindex])/TT_SCALE
        print('manually checked start = %s    auto start = %s'%(manualchain[0],laschain))

    for xdmi  in np.arange(st,en,SPACING): 
        idmi = int(dmiFun(xdmi))
        # idmi2 = int(dmiFun(xdmi+SPACING))
        # idmi2 = min(idmi2,len(dmichain-2))
        # dmi = icc.ppgps.dmi[idmi]
        lat = icc.ppgps.lat[idmi]
        lon = icc.ppgps.lon[idmi]
        elv = icc.ppgps.elev[idmi]
        # if dmi>icc.refrst and dmi<icc.secend:
        if True:
            if FINDPMARKS:
                ee,nn = pj(lon,lat)
                distrs,rsindex = treeRS.query([ee,nn])
                distls,lsindex = treeLS.query([ee,nn])
                print('LS:%.3f RS:%.3f'%(distls,distrs))
                if distls>1.5 or distls<0.65:
                    if distls>1.5:
                        distls = 1.5
                    if distls<0.65:
                        distls = 0.65
                if distrs>3 or distrs<1.5:
                    if distrs>3:
                        distrs = 3
                    if distrs<1.5:
                        distrs = 1.5
                '''THERE IS A OFFSET DISCREPANCY BETWEEN THE LIDAR AND THE MDR LOCATION'''
                ofst = 1#Rough estimate, looks pretty close
                distrs += -ofst
                distls += ofst
                print('LS:%.3f RS:%.3f  CHANGEDTO'%(distls,distrs))
                # buff = 0.2 #Extra little buffer
                # distls += buff 
                # distrs += buff 

            # id_valid_km = las.filter_by_chainage(MIN=laschain-LONGITUDE_SEARCH,MAX=laschain+LONGITUDE_SEARCH)#Filter in the section we are looking at, multiplies by 500
            # id_valid_km = las.filter_by_chainage(MIN=kmcheck2,MAX=kmcheck2+SPACING)#Filter in the section we are looking at, multiplies by 500
            # km = kmcheck2 - OG + 27300#For val site
            km = xdmi - st
            # print('\nmdrkm = %s   laskm = %s'%(km,laschain))
            # id_valid_below_sen1_KM = id_valid_below_sen1 & id_valid_km
            # id_valid_below_sen2_KM = id_valid_below_sen2 & id_valid_km
            slopes, slopetype = getCrossSlopeTIFF(tiffList,survey_name,km,distls,distrs,lat,lon)
            dicti['Route'].append(path2icc)
            dicti['kms'].append(km/1000)
            dicti['lat'].append(lat)
            dicti['lon'].append(lon)
            dicti['elv'].append(elv)
            dicti['Compare'].append(None)
            dicti['S1Fitted_%s'%survey_name].append(slopes[0])
            dicti['S1HalvedLane_%s'%survey_name].append(slopes[1])
            dicti['S2Fitted_%s'%survey_name].append(slopes[2])
            dicti['S2HalvedLane_%s'%survey_name].append(slopes[3])
            if slopetype == 'Crown':
                dicti['Super_Elevation'].append(None)
                dicti['Crown_Slope'].append(np.mean(slopes))
            else:
                dicti['Super_Elevation'].append(np.mean(slopes))
                dicti['Crown_Slope'].append(None)
            if startChainage>endChainage:
                laschain = (laschain-SPACING)
            else:
                laschain = (laschain+SPACING)
    return dicti

def extractCrossSlopeTIFF(path2TiffFolder,path2icc):
    dicti = {}
    survey_name = os.path.basename(path2TiffFolder)[:-3]
    date_folder = os.path.basename(os.path.dirname(path2TiffFolder))
    OUTPUT_FOLDER_DATED = os.path.join(OUTPUT_FOLDER,date_folder)
    if not os.path.exists(OUTPUT_FOLDER_DATED):
        os.mkdir(OUTPUT_FOLDER_DATED)
        global SURVERY_OUTPUT_FOLDER
    SURVERY_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_DATED,survey_name)
    print(SURVERY_OUTPUT_FOLDER)
    if not os.path.exists(SURVERY_OUTPUT_FOLDER):
        os.mkdir(SURVERY_OUTPUT_FOLDER)
    print('[Processing] >> ',path2TiffFolder)

    dicti = useICCTIFF(path2TiffFolder,survey_name,dicti,path2icc)
    return dicti, survey_name

def getCrossSlopeTIFF(tiffList,survey_name,km_num,distls,distrs,lat,lon):
    slopes = []



    for tiff in tiffList:
        tiff = TIFF_CLASS.Geotiff(tiff)
        # tiff.find_rot_angle_PCA()
        values = tiff.get_value_by_latlon([lat],[lon])
        if values.any() > -9998: #We will use this TIFF 
            break
    '''TODO CREATE THE BOX HERE'''
    tiffBox = np.array(x,y,z)
    


    # outname_sen1_KM = SURVERY_OUTPUT_FOLDER+r'\%s_KM%03d_sen01.las'%(survey_name,km_num)
    # outname_sen2_KM = SURVERY_OUTPUT_FOLDER+r'\%s_KM%03d_sen02.las'%(survey_name,km_num)
    # outname_sen1_KM = SURVERY_OUTPUT_FOLDER+r'\%s_%05dm_sen01.las'%(survey_name,km_num)
    # outname_sen2_KM = SURVERY_OUTPUT_FOLDER+r'\%s_%05dm_sen02.las'%(survey_name,km_num)
    outname_sen00_KM = SURVERY_OUTPUT_FOLDER+r'\%s_%05dm_sen00.las'%(survey_name,km_num)

    outnames = [outname_sen00_KM]
    # outnames = [outname_sen1_KM]
    # valids = [id_valid_below_sen1_KM]
    for a in range(len(outnames)):#Itterate over each sensor
        outname = outnames[a]

        if not os.path.exists(outname) or True:
            
            slope, newslope, visualxy, x, z, slopetype = methodTIFF(tiffBox,distls,distrs)
            # slope, newslope, visualxy, x, z, slopetype = method2(las,valid,distls,distrs)

            print('Sensor%d  %s Fit Line Reading = %3.3f%s   Half Lane Split  = %3.3f%s'%(a+1,slopetype,slope,'%',newslope,'%'))  
            slopes.append(slope)
            slopes.append(newslope)                  
            if DOVISUALIZE:#Do a visulaize output
                visualize(outname,visualxy[0],visualxy[1],x,z,distls,distrs,slope,newslope)
            
        else:
            print('Already done: ',outname)
    return slopes, slopetype

def methodTIFF(las,distls,distrs):
    '''This method takes an average of the longitudial points ang plots on a transverse'''
    #Need something here to select points a certain distance away, get the height
    # offsets = las.lasin.tt_trans_offset_m[valid]
    # points = las.lasin.z[valid]
    
    x = list(np.arange(-6,8,0.1))#Not sure about side
    # x = list(np.arange(-distrs,distls,0.1))#Not sure about side
    z = []
    trans_X = []
    elev_Z = []
    slptypesample = []
    for xx in x:
        '''Goes back to Full, wastes time'''
        # id_valid = las.filter_by_offset(OFFSET=xx)
        # newValid = valid & id_valid
        # points = las.lasin.z[newValid]
        # y.append(np.mean(points))
        '''Just does slice'''
        # id_valid = offsets== xx*TT_SCALE
        # id_valid = abs(offsets - xx*TT_SCALE) < 1 #Not sure about this range yet...
        # id_valid = abs(offsets/TT_SCALE - xx) < 0.05 #Not sure about this range yet...
        p = points[id_valid]
        if len(p) < 1:
            z.append(nan)
            continue
        avpoint = np.mean(p)
        z.append(avpoint)

        if xx > -distrs and xx < distls:
            trans_X.append(xx)
            elev_Z.append(avpoint)
            '''Get the end points'''
            lsz = avpoint
            if len(trans_X) == 1:
                rsz = avpoint
        if 3.25 < xx < 4.25:
            slptypesample.append(avpoint)


    #Split lane in half reading
    trans_X, elev_Z, slopetype = findType(slptypesample,trans_X,elev_Z,lsz,rsz)
    halfdist = (distrs+distls)/2
    half1 = np.mean(elev_Z[:len(trans_X)//2])
    half2 = np.mean(elev_Z[len(trans_X)//2:])
    newslope = ((half2-half1)/halfdist) *100

    model = lr.fit(np.array(trans_X).reshape((-1, 1)),np.array(elev_Z))
    slope = model.coef_[0] *100
    # print('Fit Line Reading = %3.3f%s'%(slope*100,'%'))
    # print('Half Lane Split  = %3.3f%s'%(newslope*100,'%'))
    # diff = ((maxpoints - minpoints)/((distls+distrs)*2))*100
    # print('Ends %s%s'%(diff,'%'))
    visualxy = [trans_X,elev_Z]

    return slope, newslope, visualxy, x, z, slopetype


if __name__ == '__main__':
    # dicti = {'kms':[],'Compare':[],'S1Fitted':[],'S1HalvedLane':[],'S2Fitted':[],'S2HalvedLane':[]}  
    # for path2las in sorted(glob.glob(sys.argv[1])):
    # for path2las in [r"X:\LiDAR\Combines\Validation\63002v_a_vux_01.las"]:
    # for path2las in [r"X:\LiDAR\Combines\20210715_7001\81702n_a_vux_01.las"]:
    # for path2las in [r"X:\LiDAR\Combines\20210920_7001\21604p3a_vux_01.las"]:
    # for path2las in glob.glob(r"X:\LiDAR\Combines\20210829_7001\85520n_v_vux_0*.las"):
    # for path2las in glob.glob(r"X:\LiDAR\Combines\20210921_7001\63002e_v_vux_*.las"):
    # for path2las in glob.glob(r"X:\LiDAR\Combines\20210910_7001\88414n_v_vux_*.las"):
    # for path2las in glob.glob(r"X:\LiDAR\Combines\20201001\01404m1a_vux_0*.laz"):
    # for path2las in glob.glob(r"Y:\Users\Trevor\crossSlope\las\01404m1a_vux_*.las"):
    # for path2las in glob.glob(r"X:\LiDAR\Combines\20210620_7001\80802s_a_vux_0*.las"):
    #     # path2icc = r'Y:\AT_2021\MDR\20210715_7001\81702n_a_icc01.mat'
    #     # pmarkshp = r"Y:\AT_2021\ASSETS\pmarks\newMethod\20220204\merged\81702N_C1R1.shp"
    #     # path2icc = r"Y:\AT_2021\Validation\20210608_63002_7001_IRI\MDR\63002e_a_icc01.mat"
    #     # path2icc = r"Y:\AT_2021\00_Validation_Collection\Data\63002\MDR_from_8th\63002e_a_icc01.mat"
    #     # path2icc = r"Y:\AT_2021\Validation\20210308_63002_7001_IRI\MDR\63002e_v_icc01.mat"
    #     # path2icc = r"Y:\2020\TRN.PAVE03181-02_SEAHD\MDR\20201001\01404m1a_icc01.mat"
    #     path2icc = r"Y:\AT_2021\MDR\20210620_7001\80802s_a_icc01.mat"
    #     dicti,survey_name = extractCrossSlope(path2las,path2icc)

    paths = [r"Y:\AT_2021\MDR\20210620_7001\80802s_a_icc01.mat"]
    for path2icc in paths:
        date = os.path.basename(os.path.dirname(path2icc))
        iccname = os.path.basename(path2icc)
        iccname = iccname.split('icc')[0]
        # for path2las in glob.glob(r"X:\LiDAR\Combines\%s\%s*.las"%(date,iccname)):
        #     dicti,survey_name = extractCrossSlope(path2las,path2icc)
        for path2TiffFolder in glob.glob(r"Y:\AT_2021\ASSETS\tiffs\TIFF_STDEV\%s\%s*"%(date,iccname)):
            dicti,survey_name = extractCrossSlopeTIFF(path2TiffFolder,path2icc)

        newdf = pd.DataFrame(dicti)
        newdf.to_csv(r'Y:\Users\Trevor\crossSlope\csvs\%s.csv'%survey_name,index=False)       