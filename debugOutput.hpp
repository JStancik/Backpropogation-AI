#pragma once
#include <iostream>
#include <string>

namespace debug{
	struct loadBar{
		std::string text;
		int segCount;
		int totalCount;
		std::string textColor;
		std::string loadColor;
		inline loadBar(std::string aText,int aSize,int aTotal,std::string aTextColor="",std::string aLoadColor=""):
		text(aText),segCount(aSize),totalCount(aTotal),textColor(aTextColor),loadColor(aLoadColor){
			std::cout<<textColor<<text<<"\t[";
			for(int i=0;i<segCount;++i){
				std::cout<<" ";
			}
			std::cout<<"]\r";
			std::cout<<text<<"\t["<<loadColor;
		}
		inline void incrLoad(int i){
			if((i+1)%(totalCount/segCount)==0){
				std::cout<<"=";
			}
		}
	};
}