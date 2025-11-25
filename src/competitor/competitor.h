#include "partition_interface.h"

template <class KEY_TYPE, class PAYLOAD_TYPE>
indexInterface<KEY_TYPE, PAYLOAD_TYPE> *get_index(std::string index_type,
                                                  std::string partition_method,
                                                  size_t &partition_num) {
  indexInterface<KEY_TYPE, PAYLOAD_TYPE> *index;
  if (partition_method == "range" || partition_method == "model" ||
      partition_method == "naive") {
    std::cout << "using partition_method: " << partition_method << std::endl;
    auto idx = new partitionedIndexInterface<KEY_TYPE, PAYLOAD_TYPE>;
    idx->prepare(index_type, partition_method, partition_num);
    index = idx;
  } else {
    if (index_type == "alex") {
      index = new alexInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "pgm") {
      index = new pgmInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "dpgm") {
      index = new dpgmInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "btree") {
      index = new BTreeInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "btreehalf") {
      index = new BTreeHalfInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "art") {
      index = new ARTInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "lipp") {
      index = new LIPPInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "sipp") {
      index = new SIPPInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "dytis") {
      index = new dytisInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "dili") {
      index = new diliInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "btreeolc") {
      index = new BTreeOLCInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "artolc") {
      index = new ARTOLCInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "xindex") {
      index = new xindexInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "finedex") {
      index = new finedexInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "finedexosm") {
      index = new finedexosmInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "alexolc") {
      index = new alexolcInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "alexol") {
      index = new alexolInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "sali") {
      index = new SALIInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "lippolc") {
      index = new LIPPOLCInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "lippol") {
      index = new LIPPOLInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "masstree") {
      index = new MasstreeInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "vega") {
      index = new VEGAInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "rs") {
      index = new RSInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else if (index_type == "rmi") {
      index = new RMIInterface<KEY_TYPE, PAYLOAD_TYPE>;
    } else {
      std::cout << "Could not find a matching index called " << index_type
                << ".\n";
      exit(0);
    }
  }
  return index;
}