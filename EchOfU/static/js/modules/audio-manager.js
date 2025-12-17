/**
 * 音频管理模块
 * 处理克隆音频列表的加载和管理
 */
class AudioManager {
    constructor() {
        this.cachedAudios = null;
        this.cacheTimestamp = null;
        this.CACHE_DURATION = 5 * 60 * 1000; // 5分钟缓存
    }

    /**
     * 从后端获取克隆音频列表
     * @param {boolean} forceRefresh - 是否强制刷新缓存
     * @returns {Promise<Array>} 克隆音频列表
     */
    async getClonedAudioList(forceRefresh = false) {
        try {
            // 检查缓存
            if (!forceRefresh && this.isCacheValid()) {
                return this.cachedAudios;
            }

            const response = await fetch('/api/cloned-audios');
            const result = await response.json();

            if (result.status === 'success' && result.audios) {
                this.cachedAudios = result.audios;
                this.cacheTimestamp = Date.now();
                return result.audios;
            } else {
                throw new Error(result.message || '获取音频列表失败');
            }

        } catch (error) {
            console.error('加载克隆音频列表失败:', error);

            // 返回默认选项
            return this.getDefaultAudioList();
        }
    }

    /**
     * 检查缓存是否有效
     * @returns {boolean}
     */
    isCacheValid() {
        return this.cachedAudios &&
               this.cacheTimestamp &&
               (Date.now() - this.cacheTimestamp) < this.CACHE_DURATION;
    }

    /**
     * 获取默认音频列表
     * @returns {Array}
     */
    getDefaultAudioList() {
        return [
            { id: 'audio_001', name: 'audio_001', created_at: '2025-01-17T10:30:00Z' },
            { id: 'audio_002', name: 'audio_002', created_at: '2025-01-17T11:15:00Z' },
            { id: 'test_audio_01', name: 'test_audio_01', created_at: '2025-01-17T09:45:00Z' },
            { id: 'sample_voice', name: 'sample_voice', created_at: '2025-01-17T08:20:00Z' }
        ];
    }

    /**
     * 将音频列表填充到选择框
     * @param {HTMLSelectElement} selectElement - 选择框元素
     * @param {string} defaultText - 默认选项文本
     * @param {string} defaultValue - 默认选项值
     * @param {boolean} forceRefresh - 是否强制刷新缓存
     */
    async populateAudioSelect(selectElement, defaultText = '请选择克隆音频', defaultValue = '', forceRefresh = false) {
        try {
            const audios = await this.getClonedAudioList(forceRefresh);

            // 清空现有选项
            selectElement.innerHTML = '';

            // 添加默认选项
            const defaultOption = document.createElement('option');
            defaultOption.value = defaultValue;
            defaultOption.textContent = defaultText;
            selectElement.appendChild(defaultOption);

            // 添加音频选项
            audios.forEach(audio => {
                const option = document.createElement('option');
                option.value = audio.id;
                option.textContent = audio.id;
                option.setAttribute('data-name', audio.name);
                option.setAttribute('data-created', audio.created_at);
                selectElement.appendChild(option);
            });

            // 添加变化事件监听器
            selectElement.addEventListener('change', (e) => {
                this.onAudioSelectChange(e.target);
            });

            return audios;
        } catch (error) {
            console.error('填充音频选择框失败:', error);
            throw error;
        }
    }

    /**
     * 音频选择框变化处理
     * @param {HTMLSelectElement} selectElement - 选择框元素
     */
    onAudioSelectChange(selectElement) {
        const selectedOption = selectElement.options[selectElement.selectedIndex];
        const audioId = selectedOption.value;

        // 触发自定义事件
        const event = new CustomEvent('audioSelected', {
            detail: {
                audioId: audioId,
                audioName: selectedOption.getAttribute('data-name') || audioId,
                created_at: selectedOption.getAttribute('data-created') || null,
                selectElement: selectElement
            }
        });

        document.dispatchEvent(event);
    }

    /**
     * 清除缓存
     */
    clearCache() {
        this.cachedAudios = null;
        this.cacheTimestamp = null;
    }

    /**
     * 刷新音频列表
     * @returns {Promise<Array>} 最新的音频列表
     */
    async refreshAudioList() {
        this.clearCache();
        return await this.getClonedAudioList();
    }

    /**
     * 根据ID获取音频信息
     * @param {string} audioId - 音频ID
     * @returns {Promise<Object|null>} 音频信息
     */
    async getAudioById(audioId) {
        try {
            const audios = await this.getClonedAudioList();
            return audios.find(audio => audio.id === audioId) || null;
        } catch (error) {
            console.error('获取音频信息失败:', error);
            return null;
        }
    }

    /**
     * 检查音频是否存在
     * @param {string} audioId - 音频ID
     * @returns {Promise<boolean>} 是否存在
     */
    async audioExists(audioId) {
        const audio = await this.getAudioById(audioId);
        return audio !== null;
    }
}

// 导出模块
window.AudioManager = AudioManager;

// 创建全局实例
window.audioManager = new AudioManager();

// 导出工具函数
window.AudioUtils = {
    /**
     * 快速填充选择框
     * @param {string} selector - 选择器
     * @param {string} defaultText - 默认文本
     * @param {string} defaultValue - 默认值
     * @param {boolean} forceRefresh - 是否强制刷新缓存
     */
    async fillAudioSelect(selector, defaultText = '请选择克隆音频', defaultValue = '', forceRefresh = false) {
        const element = document.querySelector(selector);
        if (element && element.tagName === 'SELECT') {
            return await window.audioManager.populateAudioSelect(element, defaultText, defaultValue, forceRefresh);
        }
        throw new Error('无效的选择器或元素');
    },

    /**
     * 强制刷新指定选择框的音频列表
     * @param {string} selector - 选择器
     * @param {string} defaultText - 默认文本
     * @param {string} defaultValue - 默认值
     */
    async refreshAudioSelect(selector, defaultText = '请选择克隆音频', defaultValue = '') {
        return await this.fillAudioSelect(selector, defaultText, defaultValue, true);
    },

    /**
     * 强制刷新所有音频选择框
     */
    async refreshAllAudioSelects() {
        const selects = document.querySelectorAll('select');
        const refreshPromises = [];

        selects.forEach(select => {
            // 检查是否是音频相关的选择框（根据选项内容判断）
            const hasAudioOption = Array.from(select.options).some(option =>
                option.value && option.value.includes('audio')
            );

            if (hasAudioOption) {
                const defaultOption = select.options[0];
                const defaultText = defaultOption ? defaultOption.textContent : '请选择克隆音频';
                const defaultValue = defaultOption ? defaultOption.value : '';

                refreshPromises.push(
                    this.refreshAudioSelect(`#${select.id || select.name}`, defaultText, defaultValue)
                        .catch(error => console.warn(`刷新选择框 ${select.id || select.name} 失败:`, error))
                );
            }
        });

        await Promise.all(refreshPromises);
    },

    /**
     * 获取最新的音频列表
     */
    async getLatestAudios() {
        return await window.audioManager.refreshAudioList();
    },

    /**
     * 检查音频是否存在
     * @param {string} audioId - 音频ID
     */
    async checkAudioExists(audioId) {
        return await window.audioManager.audioExists(audioId);
    },

    /**
     * 根据ID获取音频信息
     * @param {string} audioId - 音频ID
     */
    async getAudioInfo(audioId) {
        return await window.audioManager.getAudioById(audioId);
    }
};