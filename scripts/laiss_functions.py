from helper_functions import *
import pandas as pd
import numpy as np
import os
import annoy
from annoy import AnnoyIndex
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

def build_indexed_sample(fn='', lc_features=[], host_features=[], pca=True, save=True):
    host = True
    fn_stem = get_base_name(fn)

    if len(lc_features) < 1:
        print("Error! Must select at least one LC feature.")
        sys.exit()
    if len(host_features) < 1:
        host = False
    lc_and_host_features = lc_features + host_features
    data = pd.read_csv(fn, compression='gzip')
    data = data.set_index('ztf_object_id')
    if host:
        data = data[lc_and_host_features]
    else:
        data = data[lc_features]

    data = data.dropna()

    # LC + host features annoy index, w/ PCA
    feat_arr = np.array(data)
    idx_arr = np.array(data.index)

    if pca:
        scaler = preprocessing.StandardScaler()

        # Set a random seed for PCA
        random_seed = 42  # Choose your desired random seed

        # Scale the features
        feat_arr_scaled = scaler.fit_transform(feat_arr)

        # Initialize PCA with 60 principal components
        n_components = 60
        pcaModel = PCA(n_components=n_components, random_state=random_seed)

        # Apply PCA
        feat_arr_scaled_pca = pcaModel.fit_transform(feat_arr_scaled)

    # Create or load the ANNOY index
    index_nm = f"{fn_stem}_pca{pca}_host{host}_annoy_index"
    if save:
        # Save the index array to a binary file
        np.save(f"../data/{index_nm}_idx_arr.npy", idx_arr)
        np.save(f"../data/{index_nm}_feat_arr.npy", feat_arr)
        if pca:
            np.save(f"../data/{index_nm}_feat_arr_scaled.npy", feat_arr_scaled)
            np.save(f"../data/{index_nm}_feat_arr_scaled_pca.npy", feat_arr_scaled_pca)

    # Create or load the ANNOY index
    index_file = f"../data/{index_nm}.ann"  # Choose a filename
    if pca:
        index_dim = feat_arr_scaled_pca.shape[1]
    else:
        index_dim = feat_arr.shape[1]  # Dimension of the index

    # Check if the index file exists
    if not os.path.exists(index_file):
        print("Saving new ANNOY index")
        # If the index file doesn't exist, create and build the index
        index = annoy.AnnoyIndex(index_dim, metric='manhattan')

        # Add items to the index
        for i in range(len(idx_arr)):
            if pca:
                index.add_item(i, feat_arr_scaled_pca[i])
            else:
                index.add_item(i, feat_arr[i])
        # Build the index
        index.build(1000)  # 1000 trees

        if save:
            # Save the index to a file
            index.save(index_file)
    else:
        print("Loading previously saved ANNOY index")
        # If the index file exists, load it
        index = annoy.AnnoyIndex(index_dim, metric='manhattan')
        index.load(index_file)
        idx_arr = np.load(f"../data/{index_nm}_idx_arr.npy", allow_pickle=True)

def LAISS(l_or_ztfid_ref, lc_features, host_features=[], n=8, use_lc_for_ann_only_bool=False, use_ysepz_phot_snana_file=False,
          show_lightcurves_grid=False, show_hosts_grid=False, run_AD_model=False, savetables=False, savefigs=False, ad_params={}):
    print("Running LAISS...")
    lc_and_host_features = lc_features + host_features
    start_time = time.time()
    ann_num = n

    if use_ysepz_phot_snana_file:
        IAU_name = input("Input the IAU (TNS) name here, like: 2023abc\t")
        print("IAU_name:", IAU_name)
        ysepz_snana_fp = f"../ysepz_snana_phot_files/{IAU_name}_data.snana.txt"
        print(f"Looking for file {ysepz_snana_fp}...")

        # Initialize variables to store the values
        ra = None
        dec = None

        # Open the file for reading
        with open(ysepz_snana_fp, 'r') as file:
            # Read lines one by one
            for line in file:
                # Check if the line starts with '#'
                if line.startswith('#'):
                    # Split the line into key and value
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        # Check if the key is 'RA' or 'DEC'
                        if key == '# RA':
                            ra = value
                        elif key == '# DEC':
                            dec = value

        SN_df = pd.read_csv(ysepz_snana_fp, comment="#", delimiter="\s+")
        SN_df = SN_df[(SN_df.FLT == "r-ZTF") | (SN_df.FLT == "g-ZTF") | (SN_df.FLT == "g") | (SN_df.FLT == "r")].reset_index(drop=True)
        SN_df["FLT"] = SN_df["FLT"].map({"g-ZTF": "g", "g": "g", "r-ZTF": "R", "r": "R"})
        SN_df = SN_df.sort_values('MJD')
        SN_df = SN_df.dropna()
        SN_df = SN_df.drop_duplicates(keep='first')
        SN_df = SN_df.drop_duplicates(subset=['MJD'], keep='first')
        print("Using S/N cut of 3...")
        SN_df = SN_df[SN_df.FLUXCAL >= 3*SN_df.FLUXCALERR] # SNR >= 3

    figure_path = f"../LAISS_run/{l_or_ztfid_ref}/figures"
    if savefigs:
        if not os.path.exists(figure_path):
            print(f"Making figures directory {figure_path}")
            os.makedirs(figure_path)

    table_path = f"../LAISS_run/{l_or_ztfid_ref}/tables"
    if savetables:
        if not os.path.exists(table_path):
            print(f"Making tables directory {table_path}")
            os.makedirs(table_path)

    needs_reextraction_for_AD = False
    l_or_ztfid_ref_in_dataset_bank = False

    host_df_ztf_id_l, host_df_ra_l, host_df_dec_l = [], [], []

    if l_or_ztfid_ref.startswith("ANT"):
        # Get locus data using antares_client
        try:
            locus = antares_client.search.get_by_id(l_or_ztfid_ref)
        except:
            print(f"Can't get locus. Check that {l_or_ztfid_ref} is a legimiate loci! Exiting...")
            return
        ztfid_ref = locus.properties['ztf_object_id']
        needs_reextraction_for_AD = True

        if 'tns_public_objects' not in locus.catalogs:
            tns_name, tns_cls, tns_z = "No TNS", "---", -99
        else:
            tns = locus.catalog_objects['tns_public_objects'][0]
            tns_name, tns_cls, tns_z = tns['name'], tns['type'], tns['redshift']
        if tns_cls == "":
            tns_cls, tns_ann_z = "---", -99

        # Extract the relevant features
        try:
            locus_feat_arr_lc = [locus.properties[f] for f in lc_features]
            locus_feat_arr_host = [locus.properties[f] for f in host_features]
            locus_feat_arr = locus_feat_arr_lc + locus_feat_arr_host
            print(locus.properties['raMean'], locus.properties['decMean'])
            print(f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={locus.properties['raMean']}+{locus.properties['decMean']}&filter=color\n")
            host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(locus.properties['raMean']), host_df_dec_l.append(locus.properties['decMean'])

        except:
            print(f"{l_or_ztfid_ref} is not in dataset_bank. Checking if made before...")
            if os.path.exists(f"../timeseries/{ztfid_ref}_timeseries.csv"):
                print(f'{ztfid_ref} is already made. Continue!\n')
            else:
                print("Re-extracting features")
                if use_ysepz_phot_snana_file:
                    print("Using YSE-PZ SNANA Photometry file...")
                    extract_lc_and_host_features_YSE_snana_format(IAU_name=IAU_name, ztf_id_ref=l_or_ztfid_ref, yse_lightcurve=SN_df,
                                                                  ra=ra, dec=dec, show_lc=False, show_host=True)
                else:
                    extract_lc_and_host_features(ztf_id_ref=ztfid_ref, use_lc_for_ann_only_bool=use_lc_for_ann_only_bool, show_lc=False, show_host=True)

            try:
                lc_and_hosts_df = pd.read_csv(f'../timeseries/{ztfid_ref}_timeseries.csv')
            except:
                print(f"couldn't feature space as func of time for {ztfid_ref}. pass.")
                return

            lc_and_hosts_df = lc_and_hosts_df.dropna()
            print(f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df['raMean']}+{lc_and_hosts_df['decMean']}&filter=color\n")
            host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(locus.properties['raMean']), host_df_dec_l.append(locus.properties['decMean'])
            #try:
            lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
            #except:
            #    print(f"{ztfid_ref} has some NaN LC features. Skip!")
            #    return

            anom_obj_df = pd.DataFrame(lc_and_hosts_df_120d.iloc[-1]).T # last row of df to test "full LC only"
            locus_feat_arr = anom_obj_df.values[0]


    elif l_or_ztfid_ref.startswith("ZTF"):
        # Assuming you have a list of feature values
        #n = n+1 # because object in dataset chooses itself as ANN=0
        ztfid_ref = l_or_ztfid_ref

        try:
            dataset_bank_orig = pd.read_csv('../data/dataset_bank_orig_5472objs.csv.gz', compression='gzip', index_col=0)
            locus_feat_arr = dataset_bank_orig.loc[ztfid_ref]#.values
            locus_feat_arr = locus_feat_arr[lc_and_host_features].values
            needs_reextraction_for_AD = True
            l_or_ztfid_ref_in_dataset_bank = True
            print(f"{l_or_ztfid_ref} is in dataset_bank")
            n = n + 1

        except:
            print(f"{l_or_ztfid_ref} is not in dataset_bank. Checking if made before...")
            if os.path.exists(f"../timeseries/{l_or_ztfid_ref}_timeseries.csv"):
                print(f'{l_or_ztfid_ref} is already made. Continue!\n')

            else:
                print("Re-extracting LC+Host features")
                #try:
                if use_ysepz_phot_snana_file:
                    print("Using YSE-PZ SNANA Photometry file...")
                    extract_lc_and_host_features_YSE_snana_format(IAU_name=IAU_name, ztf_id_ref=l_or_ztfid_ref, yse_lightcurve=SN_df,
                                                                      ra=ra, dec=dec, show_lc=False, show_host=True, host_features=host_features)
                else:
                    extract_lc_and_host_features(ztf_id_ref=ztfid_ref, use_lc_for_ann_only_bool=use_lc_for_ann_only_bool, show_lc=False, show_host=True, host_features=host_features)
                #except:
                #    print(f"Can't extract features for {ztfid_ref}. Double check this object. Exiting...")
                #    return

            try:
                lc_and_hosts_df = pd.read_csv(f'../timeseries/{l_or_ztfid_ref}_timeseries.csv')
            except:
                print(f"couldn't feature space as func of time for {l_or_ztfid_ref}. pass.")
                return

            if not use_lc_for_ann_only_bool:
                print(f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df.iloc[0]['raMean']}+{lc_and_hosts_df.iloc[0]['decMean']}&filter=color\n")
                host_df_ztf_id_l.append(ztfid_ref), host_df_ra_l.append(lc_and_hosts_df.iloc[0]['raMean']), host_df_dec_l.append(lc_and_hosts_df.iloc[0]['decMean'])
                lc_and_hosts_df = lc_and_hosts_df.dropna() # if this drops all rows, that means something is nan from a 0 or nan entry (check data file)

                #try:
                print(lc_and_hosts_df.columns.values)
                sys.exit()

                lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
                #except:
                #    print(f"{ztfid_ref} has some NaN LC features. Skip!")
                #    return

                anom_obj_df = pd.DataFrame(lc_and_hosts_df_120d.iloc[-1]).T # last row of df to test "full LC only"
                locus_feat_arr = anom_obj_df.values[0]



            if use_lc_for_ann_only_bool:
                try:
                    lc_only_df = lc_and_hosts_df.copy()
                    lc_only_df = lc_only_df.dropna()
                    lc_only_df = lc_only_df[lc_features]
                    lc_and_hosts_df_120d = lc_only_df.copy()

                    anom_obj_df = pd.DataFrame(lc_and_hosts_df_120d.iloc[-1]).T # last row of df to test "full LC only"
                    locus_feat_arr = anom_obj_df.values[0]
                except:
                    print(f"{ztfid_ref} doesn't have enough g or r obs. Skip!")
                    return


        locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztfid_ref)
        try:
            tns = locus.catalog_objects['tns_public_objects'][0]
            tns_name, tns_cls, tns_z = tns['name'], tns['type'], tns['redshift']
        except:
            tns_name, tns_cls, tns_z = "No TNS", "---", -99
        if tns_cls == "":
            tns_cls, tns_ann_z = "---", -99



    else:
        raise ValueError("Input must be a string (l or ztfid_ref) or a list of feature values")


    if not use_lc_for_ann_only_bool:
        # 1. Scale locus_feat_arr using the same scaler (Standard Scaler)
        scaler = preprocessing.StandardScaler()
        trained_PCA_feat_arr = np.load(f"../data/dataset_bank_orig_5472objs_pcaTrue_hostTrue_annoy_index_feat_arr.npy", allow_pickle=True)

        trained_PCA_feat_arr_scaled = scaler.fit_transform(trained_PCA_feat_arr) #scaler needs to be fit first to the same data as trained

        locus_feat_arr_scaled = scaler.transform([locus_feat_arr]) # scaler transform new data

        # 2. Transform the scaled locus_feat_arr using the same PCA model (60 PCs, RS=42)
        n_components = 60
        random_seed=42
        pca = PCA(n_components=n_components, random_state=random_seed)
        trained_PCA_feat_arr_scaled_pca = pca.fit_transform(trained_PCA_feat_arr_scaled) #pca needs to be fit first to the same data as trained
        locus_feat_arr_pca = pca.transform(locus_feat_arr_scaled) # pca transform  new data

        # Create or load the ANNOY index
        #index_nm = "../dataset_bank_60pca_annoy_index" #5k, 1000 trees
        #index_file = "../dataset_bank_60pca_annoy_index.ann" #5k, 1000 trees
        index_nm = "../data/dataset_bank_orig_5472objs_pcaTrue_hostTrue_annoy_index"
        index_file = index_nm + ".ann"
        index_dim = n_components  # Dimension of the PCA index

        # 3. Use the ANNOY index to find nearest neighbors
        print("Loading previously saved ANNOY LC+HOST PCA=60 index")
        print(index_file)

        index = annoy.AnnoyIndex(index_dim, metric='manhattan')
        index.load(index_file)
        idx_arr = np.load(f"{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(locus_feat_arr_pca[0], n=n, include_distances=True)
        ann_alerce_links = [f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes]
        ann_end_time = time.time()

    else:

        if l_or_ztfid_ref_in_dataset_bank:
            locus_feat_arr = locus_feat_arr[0:62]

        # 1, 2, 3. Don't use PCA at all. Just use LC features only + ANNOY index to find nearest neighbors
        # Create or load the ANNOY index

        #TODO: make argument for which file
        #index_nm = "dataset_bank_LCfeats_only_annoy_index" #5k, 1000 trees
        #index_file = "../dataset_bank_LCfeats_only_annoy_index.ann" #5k, 1000 trees

        #index_nm = "../bigbank_90k_LCfeats_only_annoy_index_100trees" #90k, 100 trees
        #index_file = "../bigbank_90k_LCfeats_only_annoy_index_100trees.ann" #90k, 100 trees

        #index_nm = "../" #90k, 1000 trees
        #index_file = "../bigbank_90k_LCfeats_only_annoy_index_1000trees.ann" #90k, 1000 trees
        index_file = "../data/loci_df_271688objects_cut_stars_and_gal_plane_pcaFalse_hostFalse_annoy_index.ann"
        index_nm = get_base_name(index_file)
        index_dim = 62  # Dimension of the index

        print("Loading previously saved ANNOY LC-only index")
        print(index_file)
        index = annoy.AnnoyIndex(index_dim, metric='manhattan')
        index.load(index_file)
        idx_arr = np.load(f"../data/{index_nm}_idx_arr.npy", allow_pickle=True)

        ann_start_time = time.time()
        ann_indexes, ann_dists = index.get_nns_by_vector(locus_feat_arr, n=n, include_distances=True)
        ann_alerce_links = [f"https://alerce.online/object/{idx_arr[i]}" for i in ann_indexes]
        ann_end_time = time.time()

    # 4. Get TNS, spec. class of ANNs
    tns_ann_names, tns_ann_classes, tns_ann_zs = [], [], []
    ann_locus_l = []
    for i in ann_indexes:
        ann_locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=idx_arr[i])
        ann_locus_l.append(ann_locus)
        try:
            ann_tns = ann_locus.catalog_objects['tns_public_objects'][0]
            tns_ann_name, tns_ann_cls, tns_ann_z = ann_tns['name'], ann_tns['type'], ann_tns['redshift']
        except:
            tns_ann_name, tns_ann_cls, tns_ann_z = "No TNS", "---", -99
        if tns_ann_cls == "":
            tns_ann_cls, tns_ann_z = "---", -99
        tns_ann_names.append(tns_ann_name), tns_ann_classes.append(tns_ann_cls), tns_ann_zs.append(tns_ann_z)
        host_df_ztf_id_l.append(idx_arr[i])


    # Print the nearest neighbors
    print("\t\t\t\t\t   ZTFID IAU_NAME SPEC Z")
    print(f"REF. : https://alerce.online/object/{ztfid_ref} {tns_name} {tns_cls} {tns_z}")

    ann_num_l = []
    for i, (al, iau_name, spec_cls, z) in enumerate(zip(ann_alerce_links, tns_ann_names, tns_ann_classes, tns_ann_zs)):
        if l_or_ztfid_ref.startswith("ZTF"):
            if i == 0: continue
            print(f"ANN={i}: {al} {iau_name} {spec_cls}, {z}")
            ann_num_l.append(i)
        else:
            print(f"ANN={i+1}: {al} {iau_name} {spec_cls} {z}")
            ann_num_l.append(i+1)

    end_time = time.time()
    ann_elapsed_time = ann_end_time - ann_start_time
    elapsed_time = end_time - start_time
    print(f"\nANN elapsed_time = {round(ann_elapsed_time, 3)} s")
    print(f"\ntotal elapsed_time = {round(elapsed_time, 3)} s\n")

    if savetables:
        print("Saving reference+ANN table...")
        if l_or_ztfid_ref_in_dataset_bank:
            ref_and_ann_df = pd.DataFrame(zip(host_df_ztf_id_l, list(range(0, n+1)), tns_ann_names, tns_ann_classes, tns_ann_zs),
                                      columns=['ZTFID', 'ANN_NUM', 'IAU_NAME', 'SPEC_CLS', 'Z'])
        else:
            ref_and_ann_df = pd.DataFrame(zip(host_df_ztf_id_l, list(range(0, n+1)), [tns_name]+tns_ann_names, [tns_cls]+tns_ann_classes, [tns_z]+tns_ann_zs),
                                      columns=['ZTFID', 'ANN_NUM', 'IAU_NAME', 'SPEC_CLS', 'Z'])
        ref_and_ann_df.to_csv(f"{table_path}/{ztfid_ref}_ann={ann_num}.csv", index=False)
        print(f"CSV saved at: {table_path}/{ztfid_ref}_ann={ann_num}.csv")


    #############
    if show_lightcurves_grid:
        print("Making a plot of stacked lightcurves...")

        if tns_z is None:
            tns_z = "None"
        elif isinstance(tns_z, float):
            tns_z = round(tns_z, 3)
        else:
            tns_z = tns_z

        if use_ysepz_phot_snana_file:
            try:
                df_ref = SN_df
            except:
                print("No timeseries data...pass!")
                pass

            fig, ax = plt.subplots(figsize=(9.5,6))

            df_ref_g = df_ref[(df_ref.FLT == 'g') & (~df_ref.MAG.isna())]
            df_ref_r = df_ref[(df_ref.FLT == 'R') & (~df_ref.MAG.isna())]

            mjd_idx_at_min_mag_r_ref = df_ref_r[['MAG']].reset_index().idxmin().MAG
            mjd_idx_at_min_mag_g_ref = df_ref_g[['MAG']].reset_index().idxmin().MAG

            ax.errorbar(x=df_ref_r.MJD-df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref], y=df_ref_r.MAG.min()-df_ref_r.MAG, yerr=df_ref_r.MAGERR, fmt='o', c='r',
                        label=f'REF: {ztfid_ref}, d=0\n{tns_name},\t{tns_cls},\tz={tns_z}')
            ax.errorbar(x=df_ref_g.MJD-df_ref_g.MJD.iloc[mjd_idx_at_min_mag_g_ref], y=df_ref_g.MAG.min()-df_ref_g.MAG, yerr=df_ref_g.MAGERR, fmt='o', c='g')

        else:
            ref_info = antares_client.search.get_by_ztf_object_id(ztf_object_id=ztfid_ref)
            try:
                df_ref = ref_info.timeseries.to_pandas()
            except:
                print("No timeseries data...pass!")
                pass


            fig, ax = plt.subplots(figsize=(9.5,6))

            df_ref_g = df_ref[(df_ref.ant_passband == 'g') & (~df_ref.ant_mag.isna())]
            df_ref_r = df_ref[(df_ref.ant_passband == 'R') & (~df_ref.ant_mag.isna())]

            mjd_idx_at_min_mag_r_ref = df_ref_r[['ant_mag']].reset_index().idxmin().ant_mag
            mjd_idx_at_min_mag_g_ref = df_ref_g[['ant_mag']].reset_index().idxmin().ant_mag


            ax.errorbar(x=df_ref_r.ant_mjd-df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref], y=df_ref_r.ant_mag.min()-df_ref_r.ant_mag, yerr=df_ref_r.ant_magerr, fmt='o', c='r',
                        label=f'REF: {ztfid_ref}, d=0\n{tns_name},\t{tns_cls},\tz={tns_z}')
            ax.errorbar(x=df_ref_g.ant_mjd-df_ref_g.ant_mjd.iloc[mjd_idx_at_min_mag_g_ref], y=df_ref_g.ant_mag.min()-df_ref_g.ant_mag, yerr=df_ref_g.ant_magerr, fmt='o', c='g')

        markers = ['s', '*', 'x', 'P', '^', 'v', 'D', '<', '>', '8', 'p', 'x']
        consts = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]

        if l_or_ztfid_ref_in_dataset_bank:
            ann_locus_l = ann_locus_l[1:]
            host_df_ztf_id_l = host_df_ztf_id_l
            ann_dists = ann_dists[1:]
            tns_ann_names = tns_ann_names[1:]
            tns_ann_classes = tns_ann_classes[1:]
            tns_ann_zs = tns_ann_zs[1:]

        for num, (l_info, ztfname, dist, iau_name, spec_cls, z) in enumerate(zip(ann_locus_l, host_df_ztf_id_l[1:], ann_dists, tns_ann_names, tns_ann_classes, tns_ann_zs)):
            try:
                alpha=0.25
                c1='darkred'
                c2='darkgreen'

                if ztfname=='ZTF21achjwus' or ztfname=='ZTF20acnznol':
                    alpha=0.75

                df_knn = l_info.timeseries.to_pandas()

                df_g = df_knn[(df_knn.ant_passband == 'g') & (~df_knn.ant_mag.isna())]
                df_r = df_knn[(df_knn.ant_passband == 'R') & (~df_knn.ant_mag.isna())]

                mjd_idx_at_min_mag_r = df_r[['ant_mag']].reset_index().idxmin().ant_mag
                mjd_idx_at_min_mag_g = df_g[['ant_mag']].reset_index().idxmin().ant_mag

                ax.errorbar(x=df_r.ant_mjd-df_r.ant_mjd.iloc[mjd_idx_at_min_mag_r], y=df_r.ant_mag.min()-df_r.ant_mag, yerr=df_r.ant_magerr,
                            fmt=markers[num], c=c1, alpha=alpha,
                            label=f'ANN={num+1}: {ztfname}, d={int(dist)}\n{iau_name},\t{spec_cls},\tz={round(z, 3)}')
                ax.errorbar(x=df_g.ant_mjd-df_g.ant_mjd.iloc[mjd_idx_at_min_mag_g], y=df_g.ant_mag.min()-df_g.ant_mag, yerr=df_g.ant_magerr,
                            fmt=markers[num], c=c2, alpha=alpha)
                #ax.text(df_ref_r.ant_mjd.iloc[-1]-df_ref_r.ant_mjd.iloc[0]+15, df_r.ant_mag[-1]-df_r.ant_mag.min(), s=f'ANN={num+1}: {has_tns_knn}   {tns_cls_knn}')

                plt.ylabel('Apparent Mag. + Constant')
                #plt.xlabel('Days of event') # make iloc[0]
                plt.xlabel('Days since peak ($r$, $g$ indep.)') # (need r, g to be same)

                if use_ysepz_phot_snana_file:
                    if df_ref_r.MJD.iloc[0]-df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref] <= 10:
                        plt.xlim((df_ref_r.MJD.iloc[0]-df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref])-20,
                                 df_ref_r.MJD.iloc[-1]-df_ref_r.MJD.iloc[0]+15)
                    else:
                        plt.xlim(2*(df_ref_r.MJD.iloc[0]-df_ref_r.MJD.iloc[mjd_idx_at_min_mag_r_ref]),
                                 df_ref_r.MJD.iloc[-1]-df_ref_r.MJD.iloc[0]+15)

                else:
                    if df_ref_r.ant_mjd.iloc[0]-df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref] <= 10:
                        plt.xlim((df_ref_r.ant_mjd.iloc[0]-df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref])-20,
                                 df_ref_r.ant_mjd.iloc[-1]-df_ref_r.ant_mjd.iloc[0]+15)
                    else:
                        plt.xlim(2*(df_ref_r.ant_mjd.iloc[0]-df_ref_r.ant_mjd.iloc[mjd_idx_at_min_mag_r_ref]),
                                 df_ref_r.ant_mjd.iloc[-1]-df_ref_r.ant_mjd.iloc[0]+15)



                plt.legend(frameon=False,
                           loc='upper right',
                           bbox_to_anchor=(0.52, 0.85, 0.5, 0.5),
                           ncol=3,
                           columnspacing=0.75,
                           prop={'size': 12})

                plt.grid(True)

                plt.xlim(-24,107)

            except Exception as e:
                print(f"Something went wrong with plotting {ztfname}! Error is {e}. Continue...")

        if savefigs:
            print("Saving stacked lightcurve...")
            plt.savefig(f'{figure_path}/{ztfid_ref}_stacked_lightcurve_ann={ann_num}.pdf', dpi=300, bbox_inches='tight')
            print(f"PDF saved at: {figure_path}/{ztfid_ref}_stacked_lightcurve_ann={ann_num}.pdf")

        plt.show()



    ##############
    if show_hosts_grid:
        print("\nGenerating hosts grid plot...")

        dataset_bank_orig_w_hosts_ra_dec = pd.read_csv('../data/dataset_bank_orig_w_hosts_ra_dec_5472objs.csv.gz', compression='gzip', index_col=0)
        for j, ztfid in enumerate(host_df_ztf_id_l): # first entry is reference, which we already calculated
            if j == 0:
                try:
                    print(f"REF.  ({ztfid}): http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={host_df_ra_l[0]}+{host_df_dec_l[0]}&filter=color")
                    continue
                except:
                    print(f"REF.  ({ztfid}): http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].raMean}+{dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].decMean}&filter=color")
                    pass
            h_ra, h_dec = dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].raMean, dataset_bank_orig_w_hosts_ra_dec.loc[ztfid].decMean
            host_df_ra_l.append(h_ra), host_df_dec_l.append(h_dec)
            if j == 0: continue
            print(f"ANN={j} ({ztfid}): http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={h_ra}+{h_dec}&filter=color")
        host_5ann_df = pd.DataFrame(zip(host_df_ztf_id_l, host_df_ra_l, host_df_dec_l), columns=['ZTFID', 'HOST_RA', 'HOST_DEC'])
        if savefigs:
            print("Saving host thumbnails pdf...")
            host_pdfs(ztfid_ref=ztfid_ref, df=host_5ann_df, figure_path=figure_path, ann_num=ann_num, save_pdf=True)
        else:
            host_pdfs(ztfid_ref=ztfid_ref, df=host_5ann_df, figure_path=figure_path, ann_num=ann_num, save_pdf=False)

        if savetables:
            print("Saving host thumbnails table...")
            host_5ann_df.to_csv(f"{table_path}/{ztfid_ref}_host_thumbnails_ann={ann_num}.csv", index=False)
            print(f"CSV saved at: {table_path}/{ztfid_ref}_host_thumbnails_ann={ann_num}.csv")

    ########################
    if run_AD_model:
        n_estimators = ad_params['n_estimators']
        max_depth = ad_params['max_depth']
        random_state = ad_params['random_state']
        max_features = ad_params['max_features']

        figure_path = f"../models/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/figures"
        model_path = f"../models/SMOTE_train_test_70-30_min14_kneighbors8/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced/model"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        with open(f'{model_path}/cls=binary_n_estimators={n_estimators}_max_depth={max_depth}_rs={random_state}_max_feats={max_features}_cw=balanced.pkl', 'rb') as f:
            clf = pickle.load(f)

        print("\nRunning AD Model!...")
        if needs_reextraction_for_AD:
            print("Needs re-extraction for full timeseries.")
            print("Checking if made before...")
            if os.path.exists(f".../timeseries/{ztfid_ref}_timeseries.csv"):
                print(f'{ztfid_ref} is already made. Continue!\n')
            else:
                print("Re-extracting LC+HOST features")
                if use_ysepz_phot_snana_file:
                    print("Using YSE-PZ SNANA Photometry file...")
                    extract_lc_and_host_features_YSE_snana_format(IAU_name=IAU_name, ztf_id_ref=l_or_ztfid_ref, yse_lightcurve=SN_df,
                                                                  ra=ra, dec=dec, show_lc=False, show_host=True, host_features=host_features)
                else:
                    extract_lc_and_host_features(ztf_id_ref=ztfid_ref, use_lc_for_ann_only_bool=use_lc_for_ann_only_bool, show_lc=False, show_host=True, host_features=host_features)

            try:
                lc_and_hosts_df = pd.read_csv(f'../timeseries/{ztfid_ref}_timeseries.csv')
            except:
                print(f"couldn't feature space as func of time for {ztfid_ref}. pass.")
                return

            try:
                print(f"HOST : http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={lc_and_hosts_df.iloc[0]['raMean']}+{lc_and_hosts_df.iloc[0]['decMean']}&filter=color\n")
            except:
                pass

            lc_and_hosts_df = lc_and_hosts_df.dropna()

            #try:
            lc_and_hosts_df_120d = lc_and_hosts_df[lc_and_host_features]
            #except:
            #print(f"{ztfid_ref} has some NaN LC features. Skip!")

        if use_ysepz_phot_snana_file:
            plot_RFC_prob_vs_lc_yse_IAUid(clf=clf,
                                  IAU_name=IAU_name,
                                  anom_ztfid=l_or_ztfid_ref,
                                  anom_spec_cls=tns_cls,
                                  anom_spec_z=tns_z,
                                  anom_thresh=50,
                                  lc_and_hosts_df=lc_and_hosts_df,
                                  lc_and_hosts_df_120d=lc_and_hosts_df_120d,
                                  yse_lightcurve=SN_df,
                                  savefig=savefigs,
                                  figure_path=figure_path)
        else:
            plot_RFC_prob_vs_lc_ztfid(clf=clf,
                                  anom_ztfid=ztfid_ref,
                                  anom_spec_cls=tns_cls,
                                  anom_spec_z=tns_z,
                                  anom_thresh=50,
                                  lc_and_hosts_df=lc_and_hosts_df,
                                  lc_and_hosts_df_120d=lc_and_hosts_df_120d,
                                  ref_info=locus,
                                  savefig=savefigs,
                                  figure_path=figure_path)
