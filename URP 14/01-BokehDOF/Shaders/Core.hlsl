#ifndef SSGI_COMPUTE_CORE_INCLUDED
#define SSGI_COMPUTE_CORE_INCLUDED

// ================================= Macro Define ================================= //
// Helper macros for XR

#if defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_STEREO_MULTIVIEW_ENABLED)
    #define COORD_TEXTURE2D_X(pixelCoord) uint3(pixelCoord, unity_StereoEyeIndex)
    #define UNITY_XR_ASSIGN_VIEW_INDEX(viewIndex) unity_StereoEyeIndex = viewIndex
#else
    #define COORD_TEXTURE2D_X(pixelCoord) pixelCoord
    #define UNITY_XR_ASSIGN_VIEW_INDEX(viewIndex)
#endif

// Helper macros to handle XR single-pass with Texture2DArray
// With single-pass instancing, unity_StereoEyeIndex is used to select the eye in the current context.
// Otherwise, the index is statically set to 0
#if defined(USE_TEXTURE2D_X_AS_ARRAY)
    // Only single-pass stereo instancing used array indexing
    #if defined(UNITY_STEREO_INSTANCING_ENABLED)
        #define SLICE_ARRAY_INDEX   unity_StereoEyeIndex
    #else
        #define SLICE_ARRAY_INDEX  0
    #endif
#else
    #define COORD_TEXTURE2D_X(pixelCoord)                                    pixelCoord
    #define LOAD_TEXTURE2D_X_MSAA                                            LOAD_TEXTURE2D_MSAA
    #define RW_TEXTURE2D_X                                                   RW_TEXTURE2D
    #define TEXTURE2D_X_MSAA(type, textureName)                              Texture2DMS<type, 1> textureName
    #define TEXTURE2D_X_UINT(textureName)                                    Texture2D<uint> textureName
    #define TEXTURE2D_X_UINT2(textureName)                                   Texture2D<uint2> textureName
    #define INDEX_TEXTURE2D_ARRAY_X(slot)                                    (slot)
#endif
// ================================= Macro Define ================================= //
#endif