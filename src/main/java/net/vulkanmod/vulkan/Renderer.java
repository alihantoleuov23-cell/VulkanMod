package net.vulkanmod.vulkan;

import com.mojang.blaze3d.opengl.GlStateManager;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import net.minecraft.client.Minecraft;
import net.vulkanmod.Initializer;
import net.vulkanmod.gl.VkGlFramebuffer;
import net.vulkanmod.mixin.window.WindowAccessor;
import net.vulkanmod.render.shader.PipelineManager;
import net.vulkanmod.render.chunk.WorldRenderer;
import net.vulkanmod.render.chunk.buffer.UploadManager;
import net.vulkanmod.render.profiling.Profiler;
import net.vulkanmod.render.texture.ImageUploadHelper;
import net.vulkanmod.vulkan.device.DeviceManager;
import net.vulkanmod.vulkan.framebuffer.Framebuffer;
import net.vulkanmod.vulkan.framebuffer.RenderPass;
import net.vulkanmod.vulkan.framebuffer.SwapChain;
import net.vulkanmod.vulkan.memory.MemoryManager;
import net.vulkanmod.vulkan.pass.DefaultMainPass;
import net.vulkanmod.vulkan.pass.MainPass;
import net.vulkanmod.vulkan.queue.CommandPool;
import net.vulkanmod.vulkan.shader.GraphicsPipeline;
import net.vulkanmod.vulkan.shader.Pipeline;
import net.vulkanmod.vulkan.shader.PipelineState;
import net.vulkanmod.vulkan.shader.Uniforms;
import net.vulkanmod.vulkan.shader.layout.PushConstants;
import net.vulkanmod.vulkan.texture.VTextureSelector;
import net.vulkanmod.vulkan.util.VUtil;
import net.vulkanmod.vulkan.util.VkResult;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static net.vulkanmod.vulkan.Vulkan.*;
import static org.lwjgl.opengl.GL11.GL_COLOR_BUFFER_BIT;
import static org.lwjgl.opengl.GL11.GL_DEPTH_BUFFER_BIT;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.EXTDebugUtils.*;
import static org.lwjgl.vulkan.KHRSwapchain.*;
import static org.lwjgl.vulkan.VK10.*;

public class Renderer {

    private static Renderer INSTANCE;

    private static VkDevice device;

    private static boolean swapChainUpdate = false;

    public static void initRenderer() {
        INSTANCE = new Renderer();
        INSTANCE.init();
    }

    public static Renderer getInstance() {
        return INSTANCE;
    }

    public static Drawer getDrawer() {
        return INSTANCE.drawer;
    }

    public static SwapChain getSwapChain() {
        return INSTANCE.swapChain;
    }

    public static int getCurrentFrame() {
        return INSTANCE.currentFrame;
    }

    public static VkCommandBuffer getCommandBuffer() {
        return INSTANCE.currentCmdBuffer;
    }

    public static CommandPool.CommandBuffer getTransferCommandBuffer() {
        return INSTANCE.transferCbs.get(INSTANCE.currentFrame);
    }

    public static boolean isRecording() {
        return INSTANCE.recordingCmds;
    }

    private Drawer drawer;

    private SwapChain swapChain;
    private int framesNum;

    private List<VkCommandBuffer> mainCommandBuffers;

    private ArrayList<Long> imageAvailableSemaphores;
    private ArrayList<Long> renderFinishedSemaphores;
    private ArrayList<Long> inFlightFences;

    private List<CommandPool.CommandBuffer> transferCbs;

    private Framebuffer boundFramebuffer;
    private RenderPass boundRenderPass;

    private GraphicsPipeline boundPipeline;
    private long boundPipelineHandle;

    private MainPass mainPass;

    private int currentFrame = 0;
    private int currentImage = 0;
    private boolean recordingCmds = false;

    private VkCommandBuffer currentCmdBuffer;

    private final Set<Pipeline> usedPipelines = new ObjectOpenHashSet<>();
    private final List<Runnable> onResizeCallbacks = new ObjectArrayList<>();

    public Renderer() {
        device = Vulkan.getDevice();
        this.framesNum = getFramesNum();
    }

    private void init() {
        MemoryManager.createInstance(Renderer.getFramesNum());
        Vulkan.createStagingBuffers();

        swapChain = new SwapChain();
        mainPass = DefaultMainPass.create();

        drawer = new Drawer();
        drawer.createResources(framesNum);

        Uniforms.setupDefaultUniforms();
        PipelineManager.init();
        UploadManager.createInstance();

        allocateCommandBuffers();
        createSyncObjects();
    }

    private void allocateCommandBuffers() {
        if (mainCommandBuffers != null) {
            mainCommandBuffers.forEach(commandBuffer -> vkFreeCommandBuffers(device, Vulkan.getCommandPool(), commandBuffer));
        }

        mainCommandBuffers = new ArrayList<>(framesNum);

        try (MemoryStack stack = stackPush()) {
            VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack);
            allocInfo.sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
            allocInfo.commandPool(getCommandPool());
            allocInfo.level(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
            allocInfo.commandBufferCount(framesNum);

            PointerBuffer pCommandBuffers = stack.mallocPointer(framesNum);

            if (vkAllocateCommandBuffers(device, allocInfo, pCommandBuffers) != VK_SUCCESS) {
                throw new RuntimeException("Failed to allocate command buffers");
            }

            for (int i = 0; i < framesNum; i++) {
                mainCommandBuffers.add(new VkCommandBuffer(pCommandBuffers.get(i), device));
            }
        }

        if (transferCbs != null) {
            transferCbs.forEach(commandBuffer -> {
                vkResetCommandBuffer(commandBuffer.handle, 0);
                commandBuffer.reset();
            });
        }

        transferCbs = new ArrayList<>(framesNum);
        for (int i = 0; i < framesNum; i++) {
            transferCbs.add(DeviceManager.getTransferQueue().getCommandPool().getCommandBuffer());
        }
    }

    private void createSyncObjects() {
        int swapChainImages = renderFinishedSemaphores.size();
        renderFinishedSemaphores = new ArrayList<>(swapChainImages);

        imageAvailableSemaphores = new ArrayList<>(framesNum);
        inFlightFences = new ArrayList<>(framesNum);

        try (MemoryStack stack = stackPush()) {
            VkSemaphoreCreateInfo semaphoreInfo = VkSemaphoreCreateInfo.calloc(stack);
            semaphoreInfo.sType(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);

            VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack);
            fenceInfo.sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
            fenceInfo.flags(VK_FENCE_CREATE_SIGNALED_BIT);

            LongBuffer pImageAvailableSemaphore = stack.mallocLong(1);
            LongBuffer pRenderFinishedSemaphore = stack.mallocLong(1);
            LongBuffer pFence = stack.mallocLong(1);

            for (int i = 0; i < framesNum; i++) {
                if (vkCreateSemaphore(device, semaphoreInfo, null, pImageAvailableSemaphore) != VK_SUCCESS
                        || vkCreateFence(device, fenceInfo, null, pFence) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create synchronization objects for a frame");
                }

                imageAvailableSemaphores.add(pImageAvailableSemaphore.get(0));
                inFlightFences.add(pFence.get(0));
            }

            for (int i = 0; i < swapChainImages; i++) {
                if (vkCreateSemaphore(device, semaphoreInfo, null, pRenderFinishedSemaphore) != VK_SUCCESS) {
                    throw new RuntimeException("Failed to create synchronization objects for a frame");
                }

                renderFinishedSemaphores.add(pRenderFinishedSemaphore.get(0));
            }
        }
    }

    public void beginFrame() {
        beginFrame(0);
    }

    private void beginFrame(int recursion) {
        if (swapChainUpdate && recursion <= 1) {
            recreateSwapChain();
            swapChainUpdate = false;
        }

        if (swapChain.isSuboptimal() || swapChain.isResized()) {
            recreateSwapChain();
            return;
        }

        try (MemoryStack stack = stackPush()) {
            vkWaitForFences(device, inFlightFences.get(currentFrame), true, VUtil.UINT64_MAX);

            IntBuffer pImageIndex = stack.mallocInt(1);

            int vkResult = vkAcquireNextImageKHR(device, swapChain.getId(), VUtil.UINT64_MAX,
                    imageAvailableSemaphores.get(currentFrame), VK_NULL_HANDLE, pImageIndex);

            if (vkResult == VK_ERROR_OUT_OF_DATE_KHR || vkResult == VK_SUBOPTIMAL_KHR) {
                recreateSwapChain();
                return;
            } else if (vkResult != VK_SUCCESS) {
                throw new RuntimeException("Cannot acquire next image: " + VkResult.decode(vkResult));
            }

            currentImage = pImageIndex.get(0);
        }

        vkResetFences(device, inFlightFences.get(currentFrame));

        Profiler.startFrame();

        UploadManager.INSTANCE.runUploads();

        currentCmdBuffer = mainCommandBuffers.get(currentFrame);
        vkResetCommandBuffer(currentCmdBuffer, 0);

        try (MemoryStack stack = stackPush()) {
            VkCommandBufferBeginInfo beginInfo = VkCommandBufferBeginInfo.calloc(stack);
            beginInfo.sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);

            vkBeginCommandBuffer(currentCmdBuffer, beginInfo);
        }

        recordingCmds = true;

        mainPass.begin(currentCmdBuffer, stackPush());
    }

    public void endFrame() {
        Profiler.DEFAULT.push("end_frame");

        mainPass.end(currentCmdBuffer);

        vkEndCommandBuffer(currentCmdBuffer);
        recordingCmds = false;

        submitFrame();

        Profiler.DEFAULT.pop();
        Profiler.endFrame();

        cleanPasses();
    }

    public void cleanPasses() {
        usedPipelines.forEach(Pipeline::clean);
        usedPipelines.clear();
        boundPipelineHandle = 0;
        boundPipeline = null;
    }

    private void submitFrame() {
        try (MemoryStack stack = stackPush()) {
            submitUploads();

            VkSubmitInfo submitInfo = VkSubmitInfo.calloc(stack);
            submitInfo.sType(VK_STRUCTURE_TYPE_SUBMIT_INFO);

            submitInfo.waitSemaphoreCount(1);
            submitInfo.pWaitSemaphores(stack.longs(imageAvailableSemaphores.get(currentFrame)));
            submitInfo.pWaitDstStageMask(stack.ints(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT));

            submitInfo.pCommandBuffers(stack.pointers(currentCmdBuffer));

            submitInfo.pSignalSemaphores(stack.longs(renderFinishedSemaphores.get(currentImage)));

            int result = vkQueueSubmit(Vulkan.getPresentQueue().getQueue(), submitInfo, inFlightFences.get(currentFrame));

            if (result != VK_SUCCESS) {
                throw new RuntimeException("Failed to submit draw command buffer: " + VkResult.decode(result));
            }

            VkPresentInfoKHR presentInfo = VkPresentInfoKHR.calloc(stack);
            presentInfo.sType(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR);

            presentInfo.pWaitSemaphores(stack.longs(renderFinishedSemaphores.get(currentImage)));

            presentInfo.swapchainCount(1);
            presentInfo.pSwapchains(stack.longs(swapChain.getId()));

            presentInfo.pImageIndices(stack.ints(currentImage));

            result = vkQueuePresentKHR(Vulkan.getPresentQueue().getQueue(), presentInfo);

            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || swapChain.isResized()) {
                recreateSwapChain();
            } else if (result != VK_SUCCESS) {
                throw new RuntimeException("Failed to present swap chain image: " + VkResult.decode(result));
            }

            currentFrame = (currentFrame + 1) % framesNum;
        }
    }

    private void submitUploads() {
        CommandPool.CommandBuffer transferCb = transferCbs.get(currentFrame);
        if (transferCb.isSubmitted()) {
            transferCb.waitDone();
            transferCb.reset();
        }

        ImageUploadHelper.INSTANCE.submit(transferCb);

        transferCb.submit();
    }

    public void waitFences() {
        try (MemoryStack stack = stackPush()) {
            LongBuffer fences = stack.mallocLong(framesNum);
            for (int i = 0; i < framesNum; i++) {
                fences.put(i, inFlightFences.get(i));
            }

            vkWaitForFences(device, fences, true, VUtil.UINT64_MAX);
        }
    }

    public void recreateSwapChain() {
        int width = swapChain.getWidth();
        int height = swapChain.getHeight();

        while (width == 0 || height == 0) {
            width = swapChain.getWidth();
            height = swapChain.getHeight();
            Vulkan.waitIdle();
        }

        submitUploads();
        waitFences();
        Vulkan.waitIdle();

        mainCommandBuffers.forEach(commandBuffer -> vkResetCommandBuffer(commandBuffer, 0));
        recordingCmds = false;

        swapChain.recreate();

        // Semaphores need to be recreated in order to make them unsignaled
        destroySyncObjects();

        int newFramesNum = Initializer.CONFIG.frameQueueSize;

        if (framesNum != newFramesNum) {
            UploadManager.INSTANCE.submitUploads();

            framesNum = newFramesNum;
            MemoryManager.getInstance().freeAllBuffers();
            MemoryManager.createInstance(newFramesNum);
            Vulkan.createStagingBuffers();
            allocateCommandBuffers();

            Pipeline.recreateDescriptorSets(framesNum);
            drawer.createResources(framesNum);
        }

        createSyncObjects();
        this.mainPass.onResize();

        this.onResizeCallbacks.forEach(Runnable::run);
        ((WindowAccessor) (Object) Minecraft.getInstance().getWindow()).getEventHandler().resizeDisplay();

        currentFrame = 0;
    }

    public void cleanUpResources() {
        WorldRenderer.getInstance().cleanUp();
        destroySyncObjects();
        drawer.cleanUpResources();
        mainPass.cleanUp();
        swapChain.cleanUp();

        PipelineManager.destroyPipelines();
        VTextureSelector.getWhiteTexture().free();
    }

    // ИСПРАВЛЕННЫЙ МЕТОД ДЛЯ MALI GPU (БЕЗОПАСНАЯ ОЧИСТКА ВСЕХ 6 КАДРОВ БЕЗ ВЫЛЕТА):
    private void destroySyncObjects() {
        if (inFlightFences != null) {
            for (Long fence : inFlightFences) {
                if (fence != null && fence != 0L) {
                    vkDestroyFence(device, fence, null);
                }
            }
            inFlightFences.clear();
        }

        if (imageAvailableSemaphores != null) {
            for (Long sem : imageAvailableSemaphores) {
                if (sem != null && sem != 0L) {
                    vkDestroySemaphore(device, sem, null);
                }
            }
            imageAvailableSemaphores.clear();
        }

        if (renderFinishedSemaphores != null) {
            for (Long sem : renderFinishedSemaphores) {
                if (sem != null && sem != 0L) {
                    vkDestroySemaphore(device, sem, null);
                }
            }
            renderFinishedSemaphores.clear();
        }
    }

    public void addOnResizeCallback(Runnable runnable) {
        this.onResizeCallbacks.add(runnable);
    }

    public void bindGraphicsPipeline(GraphicsPipeline pipeline) {
        VkCommandBuffer commandBuffer = currentCmdBuffer;
        PipelineState currentState = PipelineState.getCurrentPipelineState(boundRenderPass);
        final long handle = pipeline.getHandle(currentState);

        if (boundPipelineHandle == handle) {
            return;
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, handle);
        boundPipelineHandle = handle;
        boundPipeline = pipeline;

        addUsedPipeline(pipeline);
    }

    public void uploadAndBindUBOs(Pipeline pipeline) {
        VkCommandBuffer commandBuffer = currentCmdBuffer;
        pipeline.bindDescriptorSets(commandBuffer, currentFrame);
    }

    public void pushConstants(Pipeline pipeline) {
        PushConstants pushConstants = pipeline.getPushConstants();
        if (pushConstants == null) return;

        try (MemoryStack stack = stackPush()) {
            ByteBuffer buffer = stack.malloc(pushConstants.getSize());
            long ptr = MemoryUtil.memAddress0(buffer);
            pushConstants.update(ptr);

            nvkCmdPushConstants(commandBuffer, pipeline.getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, pushConstants.getSize(), ptr);
        }
    }

    public Pipeline getBoundPipeline() {
        return boundPipeline;
    }

    public void setBoundFramebuffer(Framebuffer framebuffer) {
        this.boundFramebuffer = framebuffer;
    }

    public void setBoundRenderPass(RenderPass boundRenderPass) {
        this.boundRenderPass = boundRenderPass;
    }

    public void addUsedPipeline(Pipeline pipeline) {
        usedPipelines.add(pipeline);
    }

    public Framebuffer getBoundFramebuffer() {
        return boundFramebuffer;
    }

    public RenderPass getBoundRenderPass() {
        return boundRenderPass;
    }

    public MainPass getMainPass() {
        return this.mainPass;
    }

    public void setMainPass(MainPass mainPass) {
        this.mainPass = mainPass;
    }

    public CommandPool.CommandBuffer getTransferCb() {
        return transferCbs.get(currentFrame);
    }

    private static void resetDynamicState(VkCommandBuffer commandBuffer) {
        vkCmdSetDepthBias(commandBuffer, 0.0F, 0.0F, 0.0F);

        vkCmdSetLineWidth(commandBuffer, 1.0F);
    }

    public static void setDepthBias(float constant, float slope) {
        VkCommandBuffer commandBuffer = INSTANCE.currentCmdBuffer;

        vkCmdSetDepthBias(commandBuffer, constant, 0.0f, slope);
    }

    public static void clearAttachments(int attachments) {
        clearAttachments(INSTANCE.currentCmdBuffer, attachments);
    }

    public static void clearAttachments(VkCommandBuffer commandBuffer, int attachments) {
        Framebuffer framebuffer = Renderer.getInstance().boundFramebuffer;
        if (framebuffer == null)
            return;

        clearAttachments(commandBuffer, attachments, framebuffer.getWidth(), framebuffer.getHeight());
    }

    public static void clearAttachments(int attachments, int width, int height) {
        clearAttachments(INSTANCE.currentCmdBuffer, attachments, width , height);
    }

    public static void clearAttachments(int attachments, int x, int y, int width, int height) {
        clearAttachments(INSTANCE.currentCmdBuffer, attachments, x, y, width , height);
    }

    public static void clearAttachments(VkCommandBuffer commandBuffer, int attachments, int width, int height) {
        clearAttachments(commandBuffer, attachments, 0, 0, width, height);
    }

    public static void clearAttachments(VkCommandBuffer commandBuffer, int attachments, int x, int y, int width, int height) {
        try (MemoryStack stack = stackPush()) {
            VkClearValue colorValue = VkClearValue.calloc(stack);
            colorValue.color().float32(VRenderSystem.clearColor);

            VkClearValue depthValue = VkClearValue.calloc(stack);
            depthValue.depthStencil().set(VRenderSystem.clearDepthValue, 0);

            int attachmentsCount = attachments == (GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT) ? 2 : 1;
            final VkClearAttachment.Buffer pAttachments = VkClearAttachment.malloc(attachmentsCount, stack);

            switch (attachments) {
                case GL_DEPTH_BUFFER_BIT -> {
                    VkClearAttachment clearDepth = pAttachments.get(0);
                    clearDepth.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT);
                    clearDepth.colorAttachment(0);
                    clearDepth.clearValue(depthValue);
                }
                case GL_COLOR_BUFFER_BIT -> {
                    VkClearAttachment clearColor = pAttachments.get(0);
                    clearColor.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT);
                    clearColor.colorAttachment(0);
                    clearColor.clearValue(colorValue);
                }
                case GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT -> {
                    VkClearAttachment clearColor = pAttachments.get(0);
                    clearColor.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT);
                    clearColor.colorAttachment(0);
                    clearColor.clearValue(colorValue);

                    VkClearAttachment clearDepth = pAttachments.get(1);
                    clearDepth.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT);
                    clearDepth.colorAttachment(0);
                    clearDepth.clearValue(depthValue);
                }
                default -> throw new RuntimeException("unexpected value");
            }

            VkRect2D renderArea = VkRect2D.malloc(stack);
            renderArea.offset().set(x, y);
            renderArea.extent().set(width, height);

            VkClearRect.Buffer pRect = VkClearRect.malloc(1, stack);
            pRect.rect(renderArea);
            pRect.baseArrayLayer(0);
            pRect.layerCount(1);

            vkCmdClearAttachments(commandBuffer, pAttachments, pRect);
        }
    }

    public static void setInvertedViewport(int x, int y, int width, int height) {
        setViewportState(x, y + height, width, -height);
    }

    public static void resetViewport() {
        Framebuffer framebuffer = INSTANCE.getMainPass().getMainFramebuffer();

        if (framebuffer == null) {
            return;
        }

        int width = framebuffer.getWidth();
        int height = framebuffer.getHeight();

        if (width > 0 && height > 0) {
            setViewportState(0, 0, width, height);
        }
    }

    public static void setViewportState(int x, int y, int width, int height) {
        GlStateManager._viewport(x, y, width, height);
    }

    public static void setViewport(int x, int y, int width, int height) {
        try (MemoryStack stack = stackPush()) {
            setViewport(x, y, width, height, stack);
        }
    }

    public static void setViewport(int x, int y, int width, int height, MemoryStack stack) {
        if (!INSTANCE.recordingCmds)
            return;

        VkViewport.Buffer viewport = VkViewport.malloc(1, stack);
        viewport.x(x);
        viewport.y(height + y);
        viewport.width(width);
        viewport.height(-height);
        viewport.minDepth(0.0f);
        viewport.maxDepth(1.0f);

        vkCmdSetViewport(INSTANCE.currentCmdBuffer, 0, viewport);
    }

    public static void setScissor(int x, int y, int width, int height) {
        if (!INSTANCE.recordingCmds || INSTANCE.boundFramebuffer == null)
            return;

        try (MemoryStack stack = stackPush()) {
            Framebuffer framebuffer = INSTANCE.boundFramebuffer;
            int framebufferHeight = framebuffer.getHeight();

            x = Math.max(0, x);
            width = Math.min(width, framebuffer.getWidth());

            VkRect2D.Buffer scissor = VkRect2D.malloc(1, stack);
            scissor.offset().set(x, framebufferHeight - (y + height));
            scissor.extent().set(width, height);

            vkCmdSetScissor(INSTANCE.currentCmdBuffer, 0, scissor);
        }
    }

    public static void resetScissor() {
        if (INSTANCE.boundFramebuffer == null)
            return;

        try (MemoryStack stack = stackPush()) {
            VkRect2D.Buffer scissor = INSTANCE.boundFramebuffer.scissor(stack);
            vkCmdSetScissor(INSTANCE.currentCmdBuffer, 0, scissor);
        }
    }

    public static void pushDebugSection(String s) {
        Vulkan.Debug.pushDebugSection(INSTANCE.currentCmdBuffer, s);
    }

    public static void popDebugSection() {
        Vulkan.Debug.popDebugSection(INSTANCE.currentCmdBuffer);
    }

    public static int getFramesNum() {
        return INSTANCE.framesNum;
    }

    public static void scheduleSwapChainUpdate() {
        swapChainUpdate = true;
    }
}
