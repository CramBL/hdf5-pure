use hdf5_pure::AttrValue;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;

fn main() -> Result<(), Error> {
    let mut builder = FileBuilder::new();
    builder.create_group_with("run", |run| {
        run.set_attr("version", AttrValue::I32(1));
        run.create_dataset("time").with_f64_data(&[0.0, 1.0]);
        run.create_group_with("sensor", |sensor| {
            sensor.create_dataset("values").with_i32_data(&[12, 15]);
            Ok(())
        })?;
        Ok(())
    })?;
    let file = File::from_bytes(builder.finish()?)?;
    println!("{:?}", file.dataset("run/sensor/values")?.read_i32()?);
    println!("{:?}", file.group("run")?.attrs()?);
    Ok(())
}
