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
    // UTF-8 safe translations; unknown keys fall back to original text.
    zhTranslations_.clear();

    // === 基础 ===
    zhTranslations_["English"] = "English";
    zhTranslations_["Chinese"] = "中文";
    zhTranslations_["VIndex - Visual Search Engine"] = "VIndex - 视觉搜索引擎";
    zhTranslations_["Search"] = "搜索";
    zhTranslations_["Cancel"] = "取消";
    zhTranslations_["OK"] = "确定";
    zhTranslations_["Error"] = "错误";
    zhTranslations_["Warning"] = "警告";
    zhTranslations_["Success"] = "成功";
    zhTranslations_["Ready"] = "就绪";
    zhTranslations_["Close"] = "关闭";
    zhTranslations_["Clear"] = "清空";
    zhTranslations_["Browse..."] = "浏览...";
    zhTranslations_["Select"] = "选择";
    zhTranslations_["Copy"] = "复制";
    zhTranslations_["Delete"] = "删除";
    zhTranslations_["Refresh"] = "刷新";
    zhTranslations_["Save"] = "保存";
    zhTranslations_["Open"] = "打开";

    // === 菜单 ===
    zhTranslations_["&File"] = "文件(&F)";
    zhTranslations_["&Import Folder..."] = "导入文件夹(&I)...";
    zhTranslations_["E&xit"] = "退出(&X)";
    zhTranslations_["&Database"] = "数据库(&D)";
    zhTranslations_["&Rebuild Index"] = "重建索引(&R)";
    zhTranslations_["&Statistics"] = "统计信息(&S)";
    zhTranslations_["&Settings"] = "设置(&S)";
    zhTranslations_["&Preferences..."] = "首选项(&P)...";
    zhTranslations_["&Language"] = "语言(&L)";
    zhTranslations_["&Help"] = "帮助(&H)";
    zhTranslations_["&About"] = "关于(&A)";

    // === 工具栏 ===
    zhTranslations_["Main Toolbar"] = "主工具栏";
    zhTranslations_["Import Folder"] = "导入文件夹";
    zhTranslations_["Rebuild Index"] = "重建索引";

    // === 标签页 ===
    zhTranslations_["Image Search"] = "以图搜图";
    zhTranslations_["Text Search"] = "以文搜图";
    zhTranslations_["Image→Text"] = "图搜文";
    zhTranslations_["API AI"] = "API AI";
    zhTranslations_["Match"] = "图文匹配";
    zhTranslations_["Caption"] = "图像描述";
    zhTranslations_["VQA"] = "视觉问答";
    zhTranslations_["OCR"] = "文字识别";
    zhTranslations_["Library"] = "图库管理";

    // === 状态栏 ===
    zhTranslations_["Images: %1"] = "图片数: %1";
    zhTranslations_["Saving index..."] = "正在保存索引...";

    // === 模型加载 ===
    zhTranslations_["Loading models..."] = "正在加载模型...";
    zhTranslations_["Models configured successfully"] = "模型配置成功";
    zhTranslations_["Failed to load models: %1"] = "模型加载失败: %1";
    zhTranslations_["Model directory not found. Please ensure models are in ./assets/models/\n\nRun the Python export script first:\n  cd scripts && python export_clip_to_onnx.py"] = "未找到模型目录，请确保模型位于 ./assets/models/\n\n请先运行 Python 导出脚本:\n  cd scripts && python export_clip_to_onnx.py";

    // === 数据库 ===
    zhTranslations_["Database initialized successfully"] = "数据库初始化成功";
    zhTranslations_["Failed to initialize database: %1"] = "数据库初始化失败: %1";
    zhTranslations_["Database Statistics"] = "数据库统计";
    zhTranslations_["Total Images: %1"] = "总图片数: %1";
    zhTranslations_["Index Size: %1"] = "索引大小: %1";
    zhTranslations_["Categories: %1"] = "分类数: %1";
    zhTranslations_["Database Path: %1"] = "数据库路径: %1";
    zhTranslations_["Index Path: %1"] = "索引路径: %1";

    // === 导入 ===
    zhTranslations_["Select Image Folder"] = "选择图片文件夹";
    zhTranslations_["Import Options"] = "导入选项";
    zhTranslations_["Include subdirectories?"] = "是否包含子目录?";
    zhTranslations_["Importing images..."] = "正在导入图片...";
    zhTranslations_["Import Complete"] = "导入完成";
    zhTranslations_["Successfully imported %1 images"] = "成功导入 %1 张图片";
    zhTranslations_["Import failed: %1"] = "导入失败: %1";

    // === 重建索引 ===
    zhTranslations_["This will rebuild the entire search index.\nThis may take a while depending on the number of images.\n\nContinue?"] = "这将重建整个搜索索引。\n根据图片数量，可能需要一段时间。\n\n是否继续?";
    zhTranslations_["Rebuilding index..."] = "正在重建索引...";
    zhTranslations_["Index rebuilt successfully"] = "索引重建成功";
    zhTranslations_["Index rebuild completed with errors"] = "索引重建完成，但有错误";
    zhTranslations_["Rebuild failed: %1"] = "重建失败: %1";

    // === 设置/关于 ===
    zhTranslations_["Settings"] = "设置";
    zhTranslations_["Settings dialog not yet implemented.\n\nConfigure model paths in code or via config file."] = "设置对话框尚未实现。\n\n请在代码或配置文件中配置模型路径。";
    zhTranslations_["About VIndex"] = "关于 VIndex";

    // === 搜索相关 ===
    zhTranslations_["Enter search text..."] = "输入搜索文本...";
    zhTranslations_["Search by text"] = "文本搜索";
    zhTranslations_["Search by image"] = "图片搜索";
    zhTranslations_["Results"] = "结果";
    zhTranslations_["No results found"] = "未找到结果";
    zhTranslations_["Searching..."] = "搜索中...";
    zhTranslations_["Search completed"] = "搜索完成";
    zhTranslations_["Top K:"] = "返回数量:";
    zhTranslations_["Threshold:"] = "阈值:";
    zhTranslations_["Select an image to search"] = "选择图片进行搜索";
    zhTranslations_["Drop image here or click to select"] = "拖放图片到此处或点击选择";
    zhTranslations_["Select Image"] = "选择图片";
    zhTranslations_["Image Files"] = "图片文件";

    // === 图文匹配 ===
    zhTranslations_["Image"] = "图片";
    zhTranslations_["Text"] = "文本";
    zhTranslations_["Similarity Score"] = "相似度分数";
    zhTranslations_["Calculate Match"] = "计算匹配";
    zhTranslations_["Matching..."] = "匹配中...";
    zhTranslations_["Match result:"] = "匹配结果:";

    // === 图生文 ===
    zhTranslations_["Generate Caption"] = "生成描述";
    zhTranslations_["Caption:"] = "描述:";
    zhTranslations_["Generating caption..."] = "正在生成描述...";

    // === VQA ===
    zhTranslations_["Question:"] = "问题:";
    zhTranslations_["Answer:"] = "答案:";
    zhTranslations_["Ask Question"] = "提问";
    zhTranslations_["Enter your question..."] = "输入您的问题...";
    zhTranslations_["Processing..."] = "处理中...";

    // === OCR ===
    zhTranslations_["Recognize Text"] = "识别文字";
    zhTranslations_["Recognized Text:"] = "识别结果:";
    zhTranslations_["Copy Text"] = "复制文字";
    zhTranslations_["Text copied to clipboard"] = "文字已复制到剪贴板";

    // === 图库管理 ===
    zhTranslations_["Add Images"] = "添加图片";
    zhTranslations_["Remove Selected"] = "删除选中";
    zhTranslations_["Clear All"] = "清空全部";
    zhTranslations_["Category:"] = "分类:";
    zhTranslations_["All Categories"] = "全部分类";
    zhTranslations_["Confirm Delete"] = "确认删除";
    zhTranslations_["Delete selected images?"] = "删除选中的图片?";
    zhTranslations_["Confirm Clear"] = "确认清空";
    zhTranslations_["Delete ALL images from database?"] = "从数据库中删除所有图片?";
    zhTranslations_["Images deleted"] = "图片已删除";
    zhTranslations_["Database cleared"] = "数据库已清空";

    // === API AI ===
    zhTranslations_["API Settings"] = "API 设置";
    zhTranslations_["API Key:"] = "API 密钥:";
    zhTranslations_["Model:"] = "模型:";
    zhTranslations_["Send"] = "发送";
    zhTranslations_["Enter your prompt..."] = "输入您的提示词...";
    zhTranslations_["API response:"] = "API 响应:";
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
