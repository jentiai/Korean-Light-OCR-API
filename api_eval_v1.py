import os
import re
import json
import argparse

class ReadPredGT:
    def __init__(self, args):
        self.api_res_json = args.api_res_json
        self.txt_gt_dir = args.txt_gt_dir

    def read_pred_json(self):
        with open(self.api_res_json, 'r', encoding = 'UTF-8-sig') as j:
            pred_json = json.load(j)
        return pred_json

    def read_naver_pred(self, v_pred):
        self.pred_list = []
        images = v_pred["images"][0]

        fields = images["fields"]
        for field in fields:
            bounding_polys = field["boundingPoly"]["vertices"]
            inferText = field["inferText"]
            line = []
            for pts in bounding_polys:
                line.extend(map(int, [pts["x"], pts["y"]]))
            line.append(inferText)
            self.pred_list.append(line)
        
    def read_kakao_pred(self, v_pred):
        self.pred_list = []
        if "result" in v_pred.keys():
                result = v_pred["result"]
                for res in result:
                    line = []
                    for pts in res["boxes"]:
                        line.extend(map(int, pts))
                    line.append(res["recognition_words"][0])
                    self.pred_list.append(line)

    def read_jenti_pred(self, v_pred):
        self.pred_list = []
        
    def read_pred(self, v_pred):
        if 'naver' in os.path.basename(self.api_res_json):
            self.read_naver_pred(v_pred)
        elif 'kakao' in os.path.basename(self.api_res_json):
            self.read_kakao_pred(v_pred)
        elif 'jenti' in os.path.basename(self.api_res_json):
            self.read_jenti_pred(v_pred)
        return self.pred_list

    def read_txt_gt(self, k_img_name):
        gt_list = []
        p8 = re.compile(r"([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, ([\d]+)\, (.*)")
        with open(os.path.join(self.txt_gt_dir, k_img_name + '.txt'), 'r') as t:
            while True:
                line = t.readline().strip('\n').strip(' ')
                if not line:
                    break
                else:
                    m = p8.search(line)
                    det = [int(m.group(i)) for i in range(1, 9)]
                    rec = m.group(9)
                    temp = [det[0], det[1], det[2], det[3], det[4], det[5], det[6], det[7], rec]
                    gt_list.append(temp)
        return gt_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_res_json', type = str,
                        help = 'path to pred_all.json')
    parser.add_argument('--txt_gt_dir', type = str,
                        help = 'path to gt dir')
    args = parser.parse_args()

    Reader = ReadPredGT(args)    
    pred_json = Reader.read_pred_json()

    exception_list = ['B787C58869B9BE4A1A9BEF8CE75106BD', '95CC592D5B1D11F84E0BE35847F40069', '98748B55B3D715B28E75386D25B5BE29','8D4EFABEB67AFD91B582887286FB4BE6',
                        '55C08EFEA8D45CB996930A69665E93FD', 'E22BB4CDB0BCEDD60DF7442F21193A29', 'A6D141D666412FA2AAA613184AAF4030', 'F88C395F2A9A758D4B76FD0C925182F4',
                        'E315A409A419C056C63D79682BDA3771', '037AB2D5909C393AFAB7E54B5E774F90', 'B27D9FC8317B97665DAF782ADA039176', '773BD166D03EB2EA46F231C00A60FF1E',
                        '99C47D16D09A035E62CAC2C2337E5D89', 'DEB9A296C72246C51BEED525FFF24310', '2D12BB95D823D79E2F6043B6466C9E89', '143DBA4D4BD1624E09D2C4A9A47024E7',
                        '4C9ABDAD15CB0A5938A52B3622AFEE5E', 'F4A92E52BEF45770242950F0DB5CFC36', 'E60BADC7042478671400A0C215B3CE1D', '3AF424D816892489D8E8AFEAD5B7CB0E',
                        'C99E532898E0247878F3D79223D4139D', '78D19F48BB3C52DD5024E1DBA2D6FF90', '43467E673B3C90C603F4965873C31BD1', '266D58568C2CFD94FBEEF4C615945A0E',
                        '02AB20E2D013B5ABF3366ED1E0B19A59', '8E8D1AA602FA42AFD491D56DED68C70F', '3537716ED01DFDDA228B03BDD5425658', 'CA2D7FB16D23BDBFD90E04F7A1E7188B',
                        'D2B79C8606500050EE7AF1D34321E1B2', '96995710DEFFCE7717910D34CFBC5DE8', '5A0B692EF76940A7FD85BC3D47CB6771', '99E576937F56BC91AEF85150A7D4B8ED',
                        'DA02F67AF8CEA348C9B8C0864523651A', '732F76EA46A2FB72B859C55C3898A5C9', 'B6A37DBD6E2EDC079EA99D2055634D24', '672EFDF27991C2E0FE40C513A1858E43',
                        '55730FE33B59BB32DFA4E56F33EE17DF', '08A5FDE62E02BA372C87240803A22198', '01D7FCEC40BA9400C6352C5806BF3CB7', 'A0278A77DCB140E5B114E80C87C604AA',
                        '6D70446A3F49F1D0489256168187AFD4', 'F25602D844E0C05B81D5FF791D819EC3', '478C33B05B135F5FE0B6AE693E69C8D8', 'F8FCB87C09D63ED79E59E21DCFB2F5BE',
                        'F800CC17424D9D1785E032DADB8E4D99', 'D2C469551DE6599E509BCD08B721EAAD', 'BCC6067BAE7695391027C5E295E00D4E', 'D9007792B98A41059D5DAFDA9C54DF14',
                        '4E2C804F7212BC687C4A3B40BA826E1B', '9D88F0FD567E257EF46E43EC4F718753', '6E1352850CFC56AE042595CCEA37FF84', '7F97B3BF51A918798465F377FB691299',
                        '86DEF8AFEEFAAF5B77E26367DF09A0C6', 'C954D886B9F6C42DF89332D314C892F5', 'C954D886B9F6C42DF89332D314C892F5', 'A2BBA558214F315B117E6125828EA520',
                        '87124B3808B98B969595E889A417D9CF', '165C7D239F18D234E89DA9CCB350B5D0', '165C7D239F18D234E89DA9CCB350B5D0', 'D7E154EF514AF86F3F89A898902E23F7',
                        '877FE9F5CD2364A1A53AF130E4DC1A77', '27E18425BA68C427792CB892CCDDDE4E', '8EC026F0BB656DC46C0C54314197E2CD', 'E1D226CD4890911CFC5FDC0FCDD35927',
                        'AAE61BDA0C4D7B57E5D763BDBAC71B77']

    for i, (k_img_name, v_pred) in enumerate(pred_json.items()):
        if k_img_name in exception_list: # exception_list가 나오면 continue
            continue
        pred_list = Reader.read_pred(v_pred)
        gt_list = Reader.read_txt_gt(k_img_name)

        print(pred_list)
        print(gt_list)