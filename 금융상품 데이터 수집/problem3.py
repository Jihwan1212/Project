
import requests
import pprint

def get_deposit_products():
    api_key = '7be1bc61be7a7f60ee0d2d9a764792a0'
    url = f'http://finlife.fss.or.kr/finlifeapi/depositProductsSearch.json?auth={api_key}&topFinGrpNo=020000&pageNo=1'

    # API 요청 및 JSON 데이터 가져오기
    response = requests.get(url).json()
    deposit_products = response.get("result", {}).get("baseList", [])
    deposit_options = response.get("result", {}).get("optionList", [])

    return deposit_products, deposit_options

def extract_deposit_options():
    deposit_products, deposit_options = get_deposit_products()
    
    # 정제된 데이터를 저장할 리스트
    refined_list = []

    # 각 상품별 금융상품코드를 매핑하기 위한 딕셔너리 생성
    product_dict = {product["fin_prdt_cd"]: product for product in deposit_products}

    # optionList 데이터를 반복하며 필요한 정보 추출
    for option in deposit_options:
        fin_prdt_cd = option["fin_prdt_cd"]

        # 새로운 딕셔너리 생성 후 필요한 데이터 추가
        new_dict = {}
        new_dict["금융상품코드"] = fin_prdt_cd
        new_dict["저축 금리"] = option.get("intr_rate", 0)  # 기본 금리
        new_dict["저축 기간"] = option.get("save_trm", "")  # 저축 기간
        new_dict["저축금리유형"] = option.get("intr_rate_type", "")  # 금리 유형
        new_dict["저축금리유형명"] = option.get("intr_rate_type_nm", "")  # 금리 유형명

        # 최고 우대금리 추가 (option에 해당 키가 있는 경우)
        if "intr_rate2" in option:
            new_dict["최고 우대금리"] = option["intr_rate2"]

        refined_list.append(new_dict)

    return refined_list

# 실행 및 결과 출력
if __name__ == '__main__':
    result = extract_deposit_options()
    pprint.pprint(result)  # JSON 데이터를 보기 좋은 형식으로 출력