#include "translator.h"
#include <QApplication>

namespace vindex {
namespace utils {

Translator& Translator::instance() {
    static Translator instance;
    return instance;
}

Translator::Translator()
    : currentLang_(English) {
    loadTranslations();
}

void Translator::loadTranslations() {
    // ==================== 主窗口 ====================
    zhTranslations_["VIndex - Visual Search Engine"] = "VIndex - 视觉搜索引擎";
    zhTranslations_["&File"] = "文件(&F)";
    zhTranslations_["&Import Folder..."] = "导入文件夹(&I)...";
    zhTranslations_["E&xit"] = "退出(&X)";
    zhTranslations_["&Database"] = "数据库(&D)";
    zhTranslations_["&Rebuild Index"] = "重建索引(&R)";
    zhTranslations_["&Statistics"] = "统计信息(&S)";
    zhTranslations_["&Settings"] = "设置(&S)";
    zhTranslations_["&Preferences..."] = "首选项(&P)...";
    zhTranslations_["&Help"] = "帮助(&H)";
    zhTranslations_["&About"] = "关于(&A)";
    zhTranslations_["&Language"] = "语言(&L)";
    zhTranslations_["English"] = "English";
    zhTranslations_["Chinese"] = "中文";
    zhTranslations_["Main Toolbar"] = "主工具栏";
    zhTranslations_["Import Folder"] = "导入文件夹";
    zhTranslations_["Rebuild Index"] = "重建索引";
    zhTranslations_["Ready"] = "就绪";
    zhTranslations_["Images: %1"] = "图片数: %1";

    // ==================== 标签页名称 ====================
    zhTranslations_["Image Search"] = "以图搜图";
    zhTranslations_["Text Search"] = "以文搜图";
    zhTranslations_["Image→Text"] = "图搜文";
    zhTranslations_["API AI"] = "API AI";
    zhTranslations_["Match"] = "图文匹配";
    zhTranslations_["Caption"] = "图像描述";
    zhTranslations_["VQA"] = "视觉问答";
    zhTranslations_["Library"] = "图库管理";

    // ==================== 图搜图 ====================
    zhTranslations_["Query Image"] = "查询图像";
    zhTranslations_["No image selected\n\nClick 'Select Image' to choose"] = "未选择图像\n\n点击'选择图像'以选择";
    zhTranslations_["Select Image"] = "选择图像";
    zhTranslations_["Top K:"] = "结果数量:";
    zhTranslations_["Threshold:"] = "相似度阈值:";
    zhTranslations_["Search"] = "搜索";
    zhTranslations_["Search Results"] = "搜索结果";
    zhTranslations_["Select Query Image"] = "选择查询图像";
    zhTranslations_["Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"] = "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;所有文件 (*)";
    zhTranslations_["Image loaded: "] = "已加载图像: ";
    zhTranslations_["Searching..."] = "搜索中...";
    zhTranslations_["Found %1 results"] = "找到 %1 个结果";
    zhTranslations_["Please select a query image first"] = "请先选择查询图像";
    zhTranslations_["Database manager not initialized"] = "数据库管理器未初始化";
    zhTranslations_["Search failed: %1"] = "搜索失败: %1";
    zhTranslations_["Image record not found"] = "未找到图像记录";
    zhTranslations_["Error"] = "错误";

    // ==================== 文搜图 ====================
    zhTranslations_["Query Text"] = "查询文本";
    zhTranslations_["Enter search text..."] = "输入搜索文本...";
    zhTranslations_["Enter text to search for images"] = "输入文本以搜索图像";
    zhTranslations_["Text Search Results"] = "文本搜索结果";
    zhTranslations_["Please enter search text"] = "请输入搜索文本";
    zhTranslations_["Text encoder not loaded"] = "文本编码器未加载";

    // ==================== 图文匹配 ====================
    zhTranslations_["No image selected"] = "未选择图像";
    zhTranslations_["Enter text to match"] = "输入匹配文本";
    zhTranslations_["Compute Similarity"] = "计算相似度";
    zhTranslations_["Score: N/A"] = "得分: N/A";
    zhTranslations_["Score: %1"] = "得分: %1";
    zhTranslations_["Score: Error"] = "得分: 错误";
    zhTranslations_["Text:"] = "文本:";
    zhTranslations_["Please select an image"] = "请选择图像";
    zhTranslations_["Please enter text"] = "请输入文本";
    zhTranslations_["Text encoder not loaded. Please place clip_text.onnx and vocab."] = "文本编码器未加载。请放置 clip_text.onnx 和词表文件。";
    zhTranslations_["Failed to compute: %1"] = "计算失败: %1";
    zhTranslations_["Failed to load image"] = "加载图像失败";

    // ==================== 图像描述 ====================
    zhTranslations_["Input Image"] = "输入图像";
    zhTranslations_["Select an image to generate caption"] = "选择图像以生成描述";
    zhTranslations_["Generate Caption"] = "生成描述";
    zhTranslations_["Generated Caption"] = "生成的描述";
    zhTranslations_["Caption will appear here..."] = "描述将显示在这里...";
    zhTranslations_["Generating..."] = "生成中...";
    zhTranslations_["Caption model not loaded"] = "描述模型未加载";

    // ==================== 视觉问答 ====================
    zhTranslations_["Ask a question about the image"] = "询问关于图像的问题";
    zhTranslations_["Question:"] = "问题:";
    zhTranslations_["Ask"] = "提问";
    zhTranslations_["Answer"] = "答案";
    zhTranslations_["Answer will appear here..."] = "答案将显示在这里...";
    zhTranslations_["Please enter a question"] = "请输入问题";
    zhTranslations_["VQA model not loaded"] = "VQA模型未加载";

    // ==================== 图库管理 ====================
    zhTranslations_["Image Library"] = "图像库";
    zhTranslations_["Category:"] = "分类:";
    zhTranslations_["All Categories"] = "所有分类";
    zhTranslations_["Refresh"] = "刷新";
    zhTranslations_["Delete Selected"] = "删除选中";
    zhTranslations_["Clear All"] = "清空全部";
    zhTranslations_["Total: %1 images"] = "共 %1 张图像";
    zhTranslations_["Delete Images"] = "删除图像";
    zhTranslations_["Delete %1 selected images?"] = "删除选中的 %1 张图像?";
    zhTranslations_["Clear Library"] = "清空图库";
    zhTranslations_["This will delete ALL images from the library.\nThis action cannot be undone!\n\nContinue?"] = "这将删除图库中的所有图像。\n此操作无法撤销！\n\n是否继续？";
    zhTranslations_["Deleted %1 images"] = "已删除 %1 张图像";
    zhTranslations_["Library cleared"] = "图库已清空";
    zhTranslations_["No images selected"] = "未选择图像";

    // ==================== 导入对话框 ====================
    zhTranslations_["Select Image Folder"] = "选择图像文件夹";
    zhTranslations_["Import Options"] = "导入选项";
    zhTranslations_["Include subdirectories?"] = "是否包含子目录?";
    zhTranslations_["Importing images..."] = "正在导入图像...";
    zhTranslations_["Cancel"] = "取消";
    zhTranslations_["Import Complete"] = "导入完成";
    zhTranslations_["Successfully imported %1 images"] = "成功导入 %1 张图像";
    zhTranslations_["Import failed: %1"] = "导入失败: %1";

    // ==================== 重建索引 ====================
    zhTranslations_["This will rebuild the entire search index.\nThis may take a while depending on the number of images.\n\nContinue?"] = "这将重建整个搜索索引。\n根据图像数量，这可能需要一些时间。\n\n是否继续？";
    zhTranslations_["Rebuilding index..."] = "正在重建索引...";
    zhTranslations_["Success"] = "成功";
    zhTranslations_["Index rebuilt successfully"] = "索引重建成功";
    zhTranslations_["Warning"] = "警告";
    zhTranslations_["Index rebuild completed with errors"] = "索引重建完成，但有错误";
    zhTranslations_["Rebuild failed: %1"] = "重建失败: %1";

    // ==================== 统计信息 ====================
    zhTranslations_["Database Statistics"] = "数据库统计";
    zhTranslations_["Total Images: %1"] = "图像总数: %1";
    zhTranslations_["Index Size: %1"] = "索引大小: %1";
    zhTranslations_["Categories: %1"] = "分类数: %1";
    zhTranslations_["Database Path: %1"] = "数据库路径: %1";
    zhTranslations_["Index Path: %1"] = "索引路径: %1";

    // ==================== 设置对话框 ====================
    zhTranslations_["Settings"] = "设置";
    zhTranslations_["Settings dialog not yet implemented.\n\nConfigure model paths in code or via config file."] = "设置对话框尚未实现。\n\n请在代码或配置文件中设置模型路径。";

    // ==================== 关于对话框 ====================
    zhTranslations_["About VIndex"] = "关于 VIndex";

    // ==================== 模型加载 ====================
    zhTranslations_["Loading models..."] = "正在加载模型...";
    zhTranslations_["Model directory not found. Please ensure models are in ./assets/models/\n\nRun the Python export script first:\n  cd scripts && python export_clip_to_onnx.py"] = "模型目录未找到。请确保模型位于 ./assets/models/\n\n请先运行Python导出脚本:\n  cd scripts && python export_clip_to_onnx.py";
    zhTranslations_["Models configured successfully"] = "模型配置成功";
    zhTranslations_["Failed to load models: %1"] = "加载模型失败: %1";
    zhTranslations_["Database initialized successfully"] = "数据库初始化成功";
    zhTranslations_["Failed to initialize database: %1"] = "数据库初始化失败: %1";
    zhTranslations_["Saving index..."] = "正在保存索引...";

    // ==================== API AI ====================
    zhTranslations_["API Settings"] = "API 设置";
    zhTranslations_["API URL:"] = "API 地址:";
    zhTranslations_["API Key:"] = "API 密钥:";
    zhTranslations_["Model:"] = "模型:";
    zhTranslations_["Send"] = "发送";
    zhTranslations_["Response:"] = "响应:";
    zhTranslations_["Please configure API settings"] = "请配置 API 设置";

    // ==================== 图搜文 ====================
    zhTranslations_["Load Corpus"] = "加载语料";
    zhTranslations_["Corpus File:"] = "语料文件:";
    zhTranslations_["Browse..."] = "浏览...";
    zhTranslations_["Load"] = "加载";
    zhTranslations_["Corpus loaded: %1 entries"] = "语料已加载: %1 条";
    zhTranslations_["Search by Image"] = "图像搜索";
    zhTranslations_["Matched Texts"] = "匹配文本";
    zhTranslations_["Please load a corpus file first"] = "请先加载语料文件";
    zhTranslations_["Select Corpus File"] = "选择语料文件";
    zhTranslations_["Text Files (*.txt);;All Files (*)"] = "文本文件 (*.txt);;所有文件 (*)";

    // ==================== 通用 ====================
    zhTranslations_["Yes"] = "是";
    zhTranslations_["No"] = "否";
    zhTranslations_["OK"] = "确定";
    zhTranslations_["Apply"] = "应用";
    zhTranslations_["Close"] = "关闭";
    zhTranslations_["Open"] = "打开";
    zhTranslations_["Save"] = "保存";
    zhTranslations_["Copy"] = "复制";
    zhTranslations_["Paste"] = "粘贴";
    zhTranslations_["Cut"] = "剪切";
    zhTranslations_["Select All"] = "全选";
    zhTranslations_["Undo"] = "撤销";
    zhTranslations_["Redo"] = "重做";
}

void Translator::setLanguage(Language lang) {
    if (currentLang_ != lang) {
        currentLang_ = lang;
        emit languageChanged();
    }
}

QString Translator::languageName(Language lang) const {
    switch (lang) {
        case English: return "English";
        case Chinese: return "中文";
        default: return "Unknown";
    }
}

QString Translator::translate(const char* key) {
    Translator& t = instance();
    if (t.currentLang_ == English) {
        return QString::fromUtf8(key);
    }

    QString keyStr = QString::fromUtf8(key);
    auto it = t.zhTranslations_.find(keyStr);
    if (it != t.zhTranslations_.end()) {
        return it.value();
    }
    return keyStr; // 未翻译则返回原文
}

} // namespace utils
} // namespace vindex
