use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::MaxExtent;

fn main() -> Result<(), Error> {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("temperature")
        .with_unallocated_storage(hdf5_pure::make_f64_type(), &[100, 8])
        .with_chunks(&[10, 8])
        .with_maxshape(&[MaxExtent::Unlimited, MaxExtent::Fixed(8)])
        .with_fill_value(-273.15_f64);
    let file = File::from_bytes(builder.finish()?)?;
    let dataset = file.dataset("temperature")?;
    println!("Shape: {:?}", dataset.shape()?);
    println!("Allocated chunks: {}", dataset.chunks()?.len());
    println!(
        "First three values: {:?}",
        dataset.read_f64_region(&[0, 0], &[1, 3])?
    );
    Ok(())
}
