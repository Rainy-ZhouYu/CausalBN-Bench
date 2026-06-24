import pickle
import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Union
import csv
import pandas as pd
from query_helpers import questions, AttrDict, instantiate_questions, store_query_instances

dry_run = False
dataset_name = 'win95pts'
queries_path = f"./question/{dataset_name}"
# AppOK,AppData,DataFile,EMFOK,GDIIN,DS_NTOK,DS_LCLOK,LclGrbld,NtGrbld,DskLocal,PrtSpool,AppDtGnTm,PrntPrcssTm,PrtOn,PrtData,PrtStatOff,PrtPaper,PrtStatPaper,NetPrint,PC2PRT,GrbldOtpt,Problem2,PrtDriver,GDIOUT,PrtThread,DrvSet,DrvOK,PrtDataOut,PrtSel,PrtFile,PrtPath,NetOK,NtwrkCnfg,REPEAT,PrtIcon,NtSpd,PTROFFLINE,PrtCbl,LclOK,PrtPort,CblPrtHrdwrOK,DSApplctn,PrtMpTPth,PrtMem,DeskPrntSpd,CmpltPgPrntd,NnPSGrphc,PSGRAPHIC,TTOK,NnTTOK,PrtStatMem,PrtTimeOut,FllCrrptdBffr,TnrSpply,PrtStatToner,Problem1,HrglssDrtnAftrPrnt,PgOrnttnOK,PrntngArOK,ScrnFntNtPrntrFnt,IncmpltPS,Problem3,GrphcsRltdDrvrSttngs,EPSGrphc,Problem4,PrtPScript,AvlblVrtlMmry,PSERRMEM,TstpsTxt,Problem6,TrTypFnts,Problem5,FntInstlltn,PrntrAccptsTrtyp,GrbldPS,PrtQueue# CombVerMo: 综合垂直水汽含量。


variables = [
    AttrDict.make({
        "name": "AppOK",
        "expression": "Application OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "AppData",
        "expression": "Application Data",
        "singular": False,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "DataFile",
        "expression": "Data File",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "EMFOK",
        "expression": "Enhanced Metafile OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "GDIIN",
        "expression": "Graphic Device Interface Input",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "DS_NTOK",
        "expression": "NT Data Source OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
# CldShadeOth,CompPlFcst,SatContMoist,CombMoisture,RaoContMoist,VISCloudCov,CombClouds,IRCloudCover,InsInMt,AMInstabMt,OutflowFrMt,CldShadeConv,MountainFcst,WndHodograph,Boundaries,MorningBound,CapChange,InsChange,CapInScen,LoLevMoistAd,InsSclInScen,R5Fcst,Date,Scenario,ScenRelAMCIN,ScenRelAMIns,ScenRel3_4,ScnRelPlFcst,Dewpoints,LowLLapse,MeanRH,MidLLapse,MvmtFeatures,RHRatio,SfcWndShfDis,SynForcng,TempDis,WindAloft,WindFieldMt,WindFieldPln,AMCINInScen,MorningCIN,PlainsFcst,AMInsWliScen,LIfr12ZDENSd,AMDewptCalPl,N34StarFcst,LatestCIN,CurPropConv,LLIW


    AttrDict.make({
        "name": "DS_LCLOK",
        "expression": "Local Data Source OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LclGrbld",
        "expression": "Local Garbled",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "NtGrbld",
        "expression": "Windows NT system output Garbled",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "DskLocal",
        "expression": "Local Disk",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtSpool",
        "expression": "Print Spool",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "AppDtGnTm",
        "expression": "Application Data Generation Time",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrntPrcssTm",
        "expression": "Print Process Time",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "PrtOn",
        "expression": "Printer On",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtData",
        "expression": "Print Data",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtStatOff",
        "expression": "Printer Status Off",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtPaper",
        "expression": "Print Paper",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtStatPaper",
        "expression": "Printer Status Paper",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),


    AttrDict.make({
        "name": "NetPrint",
        "expression": "Network Print",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PC2PRT",
        "expression": "Data transfer from PC to printer",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "GrbldOtpt",
        "expression": "Garbled Output",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Problem2",
        "expression": "Problem 2",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "PrtDriver",
        "expression": "Printer Driver",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "GDIOUT",
        "expression": "Graphic Device Interface Output",
        "singular": False,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtThread",
        "expression": "Print Thread",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "DrvSet",
        "expression": "Driver Set",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "DrvOK",
        "expression": "Driver OK",
        "singular": False,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtDataOut",
        "expression": "Print Data Out",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "PrtSel",
        "expression": "Print Selection",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtFile",
        "expression": "Print File",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "PrtPath",
        "expression": "Print Path",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "NetOK",
        "expression": "Network OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "NtwrkCnfg",
        "expression": "Network Configuration",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "REPEAT",
        "expression": "Repeat",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtIcon",
        "expression": "Printer Icon",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "NtSpd",
        "expression": "Windows NT system speed",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PTROFFLINE",
        "expression": "Printer Offline",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "PrtCbl",
        "expression": "Printer Cable",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "LclOK",
        "expression": "Local OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtPort",
        "expression": "Printer Port",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "CblPrtHrdwrOK",
        "expression": "Cable and Printer Hardware OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "DSApplctn",
        "expression": "Data Source Application",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtMpTPth",
        "expression": "Printer Mapped Path",
        "singular": False,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "PrtMem",
        "expression": "Printer Memory",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "DeskPrntSpd",
        "expression": "Desktop Print Speed",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "CmpltPgPrntd",
        "expression": "Complete Page Printed",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "NnPSGrphc",
        "expression": "Non-PostScript Graphic",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PSGRAPHIC",
        "expression": "PostScript Graphic",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "TTOK",
        "expression": "TrueType OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "NnTTOK",
        "expression": "Non-TrueType OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "PrtStatMem",
        "expression": "Printer Status Memory",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtTimeOut",
        "expression": "Printer Timeout",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "FllCrrptdBffr",
        "expression": "Full Corrupted Buffer",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
#LatestCIN,CurPropConv,LLIW
    AttrDict.make({
        "name": "TnrSpply",
        "expression": "Printer's toner supply",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "PrtStatToner",
        "expression": "Status of the printer's toner",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Problem1",
        "expression": "Problem 1",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "HrglssDrtnAftrPrnt",
        "expression": "Hourglass Duration After Print",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PgOrnttnOK",
        "expression": "Page Orientation OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

   AttrDict.make({
        "name": "PrntngArOK",
        "expression": "Printing Area OK",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "ScrnFntNtPrntrFnt",
        "expression": "Screen Font Not Printer Font",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
#LatestCIN,CurPropConv,LLIW
    AttrDict.make({
        "name": "IncmpltPS",
        "expression": "Incomplete PostScript",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "Problem3",
        "expression": "Problem 3",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "GrphcsRltdDrvrSttngs",
        "expression": "Graphics-Related Driver Settings",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "EPSGrphc",
        "expression": "Encapsulated PostScript graphics",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Problem4",
        "expression": "Problem 4",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtPScript",
        "expression": "Print PostScript",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "AvlblVrtlMmry",
        "expression": "Available Virtual Memory",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PSERRMEM",
        "expression": "PostScript Error Memory",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "TstpsTxt",
        "expression": "Test PostScript Text",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Problem6",
        "expression": "Problem 6",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "TrTypFnts",
        "expression": "TrueType Fonts",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "Problem5",
        "expression": "Problem 5",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),

    AttrDict.make({
        "name": "FntInstlltn",
        "expression": "Font Installation",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrntrAccptsTrtyp",
        "expression": "Printer Accepts TrueType",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "GrbldPS",
        "expression": "Garbled PostScript",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
    AttrDict.make({
        "name": "PrtQueue",
        "expression": "Print Queue",
        "singular": True,
        "optionalThe": True,
        "alt": []
    }),
]

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as txt_in, open(csv_file, 'w', newline='') as csv_out:

        lines = txt_in.readlines()


        data = [line.strip().split(',') for line in lines]


        csv_writer = csv.writer(csv_out)
        csv_writer.writerows(data)
def main():
    question_instances = instantiate_questions(questions, variables)
    if not dry_run:
        store_query_instances(queries_path, question_instances)
    print("done.")
    txt_file_path = f'question/questions_{dataset_name}.txt'
    csv_file_path = f'question/questions_{dataset_name}.csv'
    txt_to_csv(txt_file_path, csv_file_path)
    df = pd.read_csv(csv_file_path, header=None)
    df.columns = ['prompt'] + df.columns.tolist()[1:]

    # 将DataFrame保存到新的csv中
    df.to_csv(f'question/questions_{dataset_name}.csv', index=False)

if __name__ == "__main__":
    main()
