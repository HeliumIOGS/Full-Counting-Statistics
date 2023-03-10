DATASETS = {#"UJ2kmk"  : "Lattice_data_kmk_uj_2",
            "UJ2" : "Lattice_data_UJ2_merged",
            #"UJ5_old" : "Lattice_data_UJ5_merged",
            "UJ5" : "Lattice_data_UJ5_merged_without_31",
            "UJ7.5" : "Lattice_data_UJ7p5",
            "UJ10" : "Lattice_data_UJ10_merged",
            #"UJ12.5_old" : "Lattice_data_UJ12p5_merged",
            #"UJ12.5" : "Lattice_data_UJ12p5_merged_without_morning_20",
            "UJ15" : "Lattice_data_UJ15_merged",
            #"UJ15kmk" : "Lattice_data_kmk_5k_fit_centering_v2",
            "UJ20" : "Lattice_data_UJ20_merged",
            "UJ22_day1" : "Lattice_data_UJ22_16032022",
            "UJ22_day2" : "Lattice_data_UJ22_17032022",
            "UJ22clean" : "Lattice_data_UJ22_300322",
            "UJ22clean2" : "Lattice_data_UJ22_070422",
            "UJ22cold" : "Lattice_data_UJ22_19042022",
            "UJ24" : "Lattice_data_UJ24",
            "UJ24_250322" : "Lattice_data_UJ24_250322",
            "UJ24_290322" : "Lattice_data_UJ24_290322",
   	    "UJ24_150622" : "Lattice_data_UJ24_150622",
   	    "UJ24_160622" : "Lattice_data_UJ24_160622",
	    "UJ24_recentered" : "Lattice_data_UJ24_recentered",
            "UJ25" : "Lattice_data_UJ25",
            "UJ26" : "Lattice_data_UJ26_14042022",
            "UJ30" : "Lattice_data_UJ30",
            "UJ30cold" : "Lattice_data_UJ30_21042022",
            "UJ35" : "Lattice_data_UJ35",
            #"MOTT": "Lattice_data_mott_high_efficiency",
            #"MOTT_HF":"Lattice_data_g2_mott_100K",
            "UJ76" : "Lattice_data_UJ76",
            "UJ100" : "Lattice_data_UJ100",		
	        "BECnoLattice" : "BEC-noLattice",
           }

SHIFT = {k:(0,0,0) for k in DATASETS.keys()}
SHIFT["MOTT"] = (-0.05, 0.05, -0.08)